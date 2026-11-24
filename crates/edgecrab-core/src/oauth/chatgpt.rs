//! OpenAI's public Sign in with ChatGPT flow for open-source local clients.
//! This is distinct from Codex device OAuth and OpenAI API-key authentication.

use std::{
    fs::{self, File, OpenOptions},
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use axum::{
    Router,
    extract::{Query, State},
    http::StatusCode,
    routing::get,
};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use chrono::Utc;
use reqwest::{Client, Url};
use ring::signature::{RSA_PKCS1_2048_8192_SHA256, RsaPublicKeyComponents};
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;
use tokio::{net::TcpListener, sync::oneshot, task, time::timeout};
use uuid::Uuid;

use super::pkce::{code_challenge, code_verifier};

pub const PROVIDER: &str = "chatgpt-pro";
const ISSUER: &str = "https://auth.openai.com";
const AUTHORIZE: &str = "https://auth.openai.com/api/accounts/authorize";
const TOKEN: &str = "https://auth.openai.com/api/accounts/oauth/token";
const REVOKE: &str = "https://auth.openai.com/api/accounts/oauth/revoke";
const JWKS: &str = "https://auth.openai.com/.well-known/jwks.json";
const RESOURCE: &str = "https://api.openai.com/v1";
const SCOPE: &str = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";
const PLAN_SCOPE: &str = "chatgpt.tokens.use.direct";
const DYNAMIC: &str = "dynamic_agent_client";
const CALLBACK: &str = "/auth/callback";
const FILENAME: &str = "chatgpt-accounts.json";
const LOGIN_COMMAND: &str = "edgecrab auth login chatgpt-pro";

#[derive(Clone, Default, Deserialize, Serialize)]
struct Store {
    host: String,
    active: Option<String>,
    accounts: Vec<Account>,
}

#[derive(Clone, Deserialize, Serialize)]
struct Account {
    client: String,
    subject: String,
    email: Option<String>,
    tokens: Option<Tokens>,
}

#[derive(Clone, Deserialize, Serialize)]
struct Tokens {
    access: String,
    refresh: String,
    id: String,
    scope: String,
    expires_at: i64,
}

impl Tokens {
    fn allows_plan(&self) -> bool {
        self.scope.split_whitespace().any(|s| s == PLAN_SCOPE)
            && self
                .scope
                .split_whitespace()
                .any(|s| s == "resource.invoke")
    }

    fn needs_refresh(&self, now: i64) -> bool {
        self.expires_at <= now.saturating_add(60)
    }
}

impl Store {
    fn active(&self) -> Option<&Account> {
        self.accounts
            .iter()
            .find(|a| Some(a.client.as_str()) == self.active.as_deref())
    }

    fn active_mut(&mut self) -> Option<&mut Account> {
        self.accounts
            .iter_mut()
            .find(|a| Some(a.client.as_str()) == self.active.as_deref())
    }

    fn replace(&mut self, account: Account) -> Result<()> {
        if let Some(existing) = self
            .accounts
            .iter_mut()
            .find(|a| a.client == account.client)
        {
            ensure!(
                existing.subject == account.subject,
                "ChatGPT sign-in returned a different account; previous credentials were kept"
            );
            self.active = Some(account.client.clone());
            *existing = account;
        } else {
            self.active = Some(account.client.clone());
            self.accounts.push(account);
        }
        Ok(())
    }
}

fn store_path() -> PathBuf {
    crate::config::edgecrab_home().join(FILENAME)
}

fn read_store(path: &Path) -> Result<Store> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|_| anyhow!("ChatGPT account store is unreadable")),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(Store::default()),
        Err(e) => Err(e).context("Cannot read ChatGPT account store"),
    }
}

fn save_store(path: &Path, store: &Store) -> Result<()> {
    let dir = path
        .parent()
        .context("ChatGPT account store directory not found")?;
    // NamedTempFile creates a 0600 file on Unix. The rename retains those permissions.
    let mut temp = tempfile::NamedTempFile::new_in(dir)
        .context("Cannot create protected ChatGPT account file")?;
    serde_json::to_writer(&mut temp, store)
        .map_err(|_| anyhow!("Cannot write ChatGPT account file"))?;
    temp.flush()?;
    temp.as_file().sync_all()?;
    temp.persist(path)
        .map_err(|e| e.error)
        .context("Cannot replace ChatGPT account store")?;
    Ok(())
}

async fn lock_store(path: &Path) -> Result<File> {
    let path = path.with_extension("lock");
    task::spawn_blocking(move || {
        fs::create_dir_all(
            path.parent()
                .context("ChatGPT account store directory not found")?,
        )?;
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let lock = options.open(path)?;
        lock.lock()?;
        Ok(lock)
    })
    .await
    .context("Cannot lock ChatGPT account store")?
}

fn client() -> Result<Client> {
    Client::builder()
        .timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| anyhow!("Cannot initialize the ChatGPT sign-in HTTP client"))
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    id_token: Option<String>,
    token_type: String,
    expires_in: i64,
    scope: Option<String>,
}

impl TokenResponse {
    fn credentials(self, prior: Option<&Tokens>, id: String) -> Result<Tokens> {
        ensure!(
            self.token_type.eq_ignore_ascii_case("bearer")
                && !self.access_token.is_empty()
                && self.expires_in > 0,
            "ChatGPT returned an unusable access token"
        );
        let refresh = self
            .refresh_token
            .or_else(|| prior.map(|p| p.refresh.clone()))
            .filter(|t| !t.is_empty());
        let scope = self
            .scope
            .or_else(|| prior.map(|p| p.scope.clone()))
            .unwrap_or_default();
        let expires_at = Utc::now()
            .timestamp()
            .checked_add(self.expires_in)
            .context("ChatGPT returned an invalid expiry")?;
        Ok(Tokens {
            access: self.access_token,
            refresh: refresh.unwrap_or_default(),
            id,
            scope,
            expires_at,
        })
    }
}

#[derive(Deserialize)]
struct OAuthErrorBody {
    error: Option<String>,
}

struct TokenError {
    status: Option<reqwest::StatusCode>,
    code: Option<String>,
}

impl TokenError {
    fn terminal_refresh(&self) -> bool {
        self.status.is_some_and(|s| s.is_client_error())
            && matches!(
                self.code.as_deref(),
                Some(
                    "invalid_grant"
                        | "invalid_refresh_token"
                        | "token_expired"
                        | "refresh_token_expired"
                        | "refresh_token_invalidated"
                        | "refresh_token_reused"
                )
            )
    }

    fn safe_message(&self) -> String {
        if let Some(status) = self.status {
            format!("ChatGPT token exchange failed (HTTP {})", status.as_u16())
        } else {
            "Could not contact the ChatGPT token endpoint; retry later".to_owned()
        }
    }
}

async fn exchange(
    http: &Client,
    form: &[(&str, &str)],
) -> std::result::Result<TokenResponse, TokenError> {
    let response = http
        .post(TOKEN)
        .form(form)
        .send()
        .await
        .map_err(|_| TokenError {
            status: None,
            code: None,
        })?;
    let status = response.status();
    if status.is_success() {
        return response.json().await.map_err(|_| TokenError {
            status: Some(status),
            code: None,
        });
    }
    let code = response
        .json::<OAuthErrorBody>()
        .await
        .ok()
        .and_then(|r| r.error);
    Err(TokenError {
        status: Some(status),
        code,
    })
}

#[derive(Deserialize)]
struct Claims {
    iss: String,
    sub: String,
    email: Option<String>,
    aud: Audience,
    azp: Option<String>,
    exp: i64,
    nbf: Option<i64>,
    nonce: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Audience {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Deserialize)]
struct JwtHeader {
    alg: String,
    kid: String,
    #[serde(default)]
    crit: Vec<String>,
}

#[derive(Deserialize)]
struct Jwks {
    keys: Vec<Jwk>,
}

#[derive(Deserialize)]
struct Jwk {
    kty: String,
    kid: String,
    alg: Option<String>,
    #[serde(rename = "use")]
    usage: Option<String>,
    n: Option<String>,
    e: Option<String>,
}

fn validate_claims(claims: &Claims, issued_client: &str, nonce: &str, now: i64) -> Result<()> {
    let audience = match &claims.aud {
        Audience::Single(aud) => aud == issued_client,
        Audience::Multiple(aud) => aud.as_slice() == [issued_client],
    };
    ensure!(
        claims.iss == ISSUER
            && !claims.sub.is_empty()
            && claims.exp > now
            && claims.nbf.is_none_or(|t| t <= now)
            && audience
            && claims.azp.as_deref().is_none_or(|azp| azp == issued_client)
            && bool::from(claims.nonce.as_bytes().ct_eq(nonce.as_bytes())),
        "ChatGPT identity did not match this sign-in attempt"
    );
    Ok(())
}

fn verify_id_token(
    token: &str,
    issued_client: &str,
    nonce: &str,
    keys: &Jwks,
    now: i64,
) -> Result<Claims> {
    let decode = |value: &str| {
        URL_SAFE_NO_PAD
            .decode(value)
            .map_err(|_| anyhow!("Invalid ChatGPT identity token"))
    };
    let (signed, signature) = token
        .rsplit_once('.')
        .context("Invalid ChatGPT identity token")?;
    let (header, claims) = signed
        .split_once('.')
        .context("Invalid ChatGPT identity token")?;
    let header: JwtHeader = serde_json::from_slice(&decode(header)?)
        .map_err(|_| anyhow!("Invalid ChatGPT identity token header"))?;
    ensure!(
        header.alg == "RS256" && header.crit.is_empty(),
        "Unsupported ChatGPT identity token algorithm"
    );
    let mut matches = keys.keys.iter().filter(|key| {
        key.kid == header.kid
            && key.kty == "RSA"
            && key.alg.as_deref().is_none_or(|alg| alg == "RS256")
            && key.usage.as_deref().is_none_or(|usage| usage == "sig")
    });
    let key = matches
        .next()
        .context("ChatGPT identity token signing key was not found")?;
    ensure!(
        matches.next().is_none(),
        "ChatGPT returned ambiguous identity signing keys"
    );
    let n = decode(key.n.as_deref().context("Invalid ChatGPT signing key")?)?;
    let e = decode(key.e.as_deref().context("Invalid ChatGPT signing key")?)?;
    RsaPublicKeyComponents { n: &n, e: &e }
        .verify(
            &RSA_PKCS1_2048_8192_SHA256,
            signed.as_bytes(),
            &decode(signature)?,
        )
        .map_err(|_| anyhow!("ChatGPT identity token signature could not be verified"))?;
    let claims = serde_json::from_slice(&decode(claims)?)
        .map_err(|_| anyhow!("Invalid ChatGPT identity claims"))?;
    validate_claims(&claims, issued_client, nonce, now)?;
    Ok(claims)
}

#[derive(Deserialize)]
struct Callback {
    state: Option<String>,
    code: Option<String>,
    client_id: Option<String>,
    error: Option<String>,
}

struct CallbackWaiter {
    state: String,
    callback: Mutex<Option<oneshot::Sender<Callback>>>,
}

async fn callback(
    State(waiter): State<Arc<CallbackWaiter>>,
    Query(value): Query<Callback>,
) -> (StatusCode, &'static str) {
    if !value
        .state
        .as_ref()
        .is_some_and(|state| bool::from(state.as_bytes().ct_eq(waiter.state.as_bytes())))
    {
        return (
            StatusCode::BAD_REQUEST,
            "This callback does not match the active EdgeCrab sign-in.",
        );
    }
    let Some(sender) = waiter
        .callback
        .lock()
        .ok()
        .and_then(|mut sender| sender.take())
    else {
        return (
            StatusCode::GONE,
            "This EdgeCrab sign-in attempt has already ended.",
        );
    };
    let _ = sender.send(value);
    (
        StatusCode::OK,
        "EdgeCrab received the sign-in response. Return to your terminal to check the result.",
    )
}

fn issued_client(callback: &Callback, returning: Option<&str>) -> Result<String> {
    ensure!(
        callback.error.is_none(),
        "ChatGPT sign-in was declined or did not finish; no credentials were changed"
    );
    if let Some(client) = returning {
        ensure!(
            callback.client_id.as_deref().is_none_or(|id| id == client),
            "ChatGPT returned a different client registration"
        );
        return Ok(client.to_owned());
    }
    callback
        .client_id
        .as_deref()
        .filter(|value| value.starts_with("oaiapp_") && value.len() > "oaiapp_".len())
        .map(str::to_owned)
        .context("ChatGPT did not return an issued client ID; registration is incomplete")
}

fn authorize_url(
    host: &str,
    client_id: Option<&str>,
    redirect: &str,
    state: &str,
    nonce: &str,
    verifier: &str,
    consent: bool,
) -> Result<Url> {
    let mut url = Url::parse(AUTHORIZE)?;
    let mut query = url.query_pairs_mut();
    query.extend_pairs([
        ("client_id", client_id.unwrap_or(DYNAMIC)),
        ("ext_agent_host_id", host),
        ("redirect_uri", redirect),
        ("response_type", "code"),
        ("scope", SCOPE),
        ("resource", RESOURCE),
        ("state", state),
        ("nonce", nonce),
        ("code_challenge_method", "S256"),
        ("code_challenge", &code_challenge(verifier)),
    ]);
    if client_id.is_none() {
        query.append_pair("agent_name_hint", "EdgeCrab");
    } else if consent {
        query.append_pair("prompt", "consent");
    }
    // Hints are optional. Never place a retained ID token in a URL shown in a terminal.
    drop(query);
    Ok(url)
}

/// Start the real browser flow. `account` chooses a saved registration (by the
/// client ID shown by `accounts`); `new_account` keeps an existing login intact
/// until this new account has successfully authenticated.
pub async fn login(
    account: Option<&str>,
    new_account: bool,
    port: u16,
    open_browser: impl FnOnce(&Url, u16),
) -> Result<()> {
    ensure!(
        !new_account || account.is_none(),
        "Choose either a saved ChatGPT account or a new one"
    );
    let path = store_path();
    let (host, returning) = {
        let _lock = lock_store(&path).await?;
        let mut store = read_store(&path)?;
        if store.host.is_empty() {
            store.host = format!("urn:uuid:{}", Uuid::new_v4());
            save_store(&path, &store)?;
        }
        let selected = match account {
            Some(client) => Some(
                store
                    .accounts
                    .iter()
                    .find(|a| a.client == client)
                    .context("Unknown saved ChatGPT account")?,
            ),
            None if new_account => None,
            None => store.active(),
        };
        let selected = selected.cloned();
        (store.host, selected)
    };
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port))
        .await
        .context(
            "Cannot open the local ChatGPT callback listener; choose an unused callback port",
        )?;
    let bound_port = listener.local_addr()?.port();
    let redirect = format!("http://127.0.0.1:{bound_port}{CALLBACK}");
    let state = code_verifier();
    let nonce = code_verifier();
    let verifier = code_verifier();
    let saved_client = returning.as_ref().map(|r| r.client.as_str());
    let consent = returning
        .as_ref()
        .and_then(|r| r.tokens.as_ref())
        .is_some_and(|t| !t.allows_plan());
    let url = authorize_url(
        &host,
        saved_client,
        &redirect,
        &state,
        &nonce,
        &verifier,
        consent,
    )?;
    let (tx, rx) = oneshot::channel();
    let app = Router::new()
        .route(CALLBACK, get(callback))
        .with_state(Arc::new(CallbackWaiter {
            state,
            callback: Mutex::new(Some(tx)),
        }));
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let server = task::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = stop_rx.await;
            })
            .await
    });
    open_browser(&url, bound_port);
    let reply = timeout(Duration::from_secs(300), rx).await;
    let _ = stop_tx.send(());
    // Do not wait for lingering browser connections.
    drop(server);
    let reply = reply
        .context("ChatGPT sign-in timed out; restart it and use the local browser")?
        .context("Local ChatGPT callback closed unexpectedly")?;
    let client_id = issued_client(&reply, saved_client)?;
    let code = reply
        .code
        .as_deref()
        .filter(|code| !code.is_empty())
        .context("ChatGPT callback did not include an authorization code")?;
    let http = client()?;
    let response = exchange(
        &http,
        &[
            ("grant_type", "authorization_code"),
            ("client_id", &client_id),
            ("code", code),
            ("code_verifier", &verifier),
            ("redirect_uri", &redirect),
            ("resource", RESOURCE),
        ],
    )
    .await
    .map_err(|e| anyhow!("{}; start a new sign-in", e.safe_message()))?;
    let id = response
        .id_token
        .clone()
        .context("ChatGPT did not return an ID token")?;
    let keys: Jwks = http
        .get(JWKS)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|_| anyhow!("Cannot retrieve OpenAI signing keys"))?
        .json()
        .await
        .map_err(|_| anyhow!("Cannot read OpenAI signing keys"))?;
    let claims = verify_id_token(&id, &client_id, &nonce, &keys, Utc::now().timestamp())?;
    ensure!(
        returning.as_ref().is_none_or(|r| r.subject == claims.sub),
        "ChatGPT sign-in returned a different account; previous credentials were kept"
    );
    let tokens = response.credentials(None, id)?;
    let _lock = lock_store(&path).await?;
    let mut store = read_store(&path)?;
    ensure!(
        store.host == host,
        "The EdgeCrab sign-in host changed; retry sign-in"
    );
    store.replace(Account {
        client: client_id,
        subject: claims.sub,
        email: claims.email,
        tokens: Some(tokens),
    })?;
    save_store(&path, &store)
}

pub struct AccountStatus {
    pub client_id: String,
    pub email: Option<String>,
    pub active: bool,
    pub status: &'static str,
}

pub fn accounts() -> Result<Vec<AccountStatus>> {
    let store = read_store(&store_path())?;
    Ok(store
        .accounts
        .into_iter()
        .map(|account| {
            let status = match &account.tokens {
                None => "signed out",
                Some(t) if !t.allows_plan() => "ChatGPT plan usage not authorized",
                Some(t) if t.needs_refresh(Utc::now().timestamp()) && t.refresh.is_empty() => {
                    "sign-in required"
                }
                Some(t) if t.needs_refresh(Utc::now().timestamp()) => {
                    "will refresh on next request"
                }
                Some(_) => "ready",
            };
            AccountStatus {
                active: Some(&account.client) == store.active.as_ref(),
                client_id: account.client,
                email: account.email,
                status,
            }
        })
        .collect())
}

/// Resolve the active account's bearer token; do not use API-key environment
/// variables or any legacy Codex credentials.
pub async fn access_token() -> Result<String> {
    let path = store_path();
    let _lock = lock_store(&path).await?;
    let mut store = read_store(&path)?;
    let account = store.active_mut().with_context(|| {
        format!("No ChatGPT subscription account is selected; run `{LOGIN_COMMAND}`")
    })?;
    let tokens = account.tokens.as_ref().with_context(|| {
        format!("Your ChatGPT subscription session is signed out; run `{LOGIN_COMMAND}`")
    })?;
    ensure!(
        tokens.allows_plan(),
        "ChatGPT plan usage is not authorized; run `{LOGIN_COMMAND}` and enable plan usage in the browser"
    );
    if !tokens.needs_refresh(Utc::now().timestamp()) {
        return Ok(tokens.access.clone());
    }
    ensure!(
        !tokens.refresh.is_empty(),
        "Your ChatGPT subscription session expired; run `{LOGIN_COMMAND}`"
    );
    let response = exchange(
        &client()?,
        &[
            ("grant_type", "refresh_token"),
            ("client_id", &account.client),
            ("refresh_token", &tokens.refresh),
            ("resource", RESOURCE),
        ],
    )
    .await;
    match response {
        Ok(response) => {
            // Retain the original, validated ID token as identity evidence; a
            // refresh response does not take part in the original nonce exchange.
            let replacement = response.credentials(Some(tokens), tokens.id.clone())?;
            let allows_plan = replacement.allows_plan();
            let access = replacement.access.clone();
            account.tokens = Some(replacement);
            save_store(&path, &store)?;
            ensure!(
                allows_plan,
                "ChatGPT plan usage is no longer authorized; run `{LOGIN_COMMAND}`"
            );
            Ok(access)
        }
        Err(error) if error.terminal_refresh() => {
            account.tokens = None;
            save_store(&path, &store)?;
            bail!(
                "Your ChatGPT subscription session is no longer renewable; run `{LOGIN_COMMAND}`"
            );
        }
        Err(error) => bail!("{}; your saved session was kept", error.safe_message()),
    }
}

/// Sign out the active account but retain its registration for a later sign-in.
/// Returns whether remote revocation was confirmed (or was unnecessary).
pub async fn logout() -> Result<bool> {
    logout_accounts(false).await
}

/// Clear all renewable sessions owned by EdgeCrab; retain their public client registrations.
pub async fn logout_all() -> Result<bool> {
    logout_accounts(true).await
}

async fn logout_accounts(all: bool) -> Result<bool> {
    let path = store_path();
    let _lock = lock_store(&path).await?;
    let mut store = read_store(&path)?;
    let mut confirmed = true;
    let mut changed = false;
    if all {
        for account in &mut store.accounts {
            if let Some(tokens) = account.tokens.take() {
                changed = true;
                confirmed &= revoke(&tokens.refresh, &account.client).await;
            }
        }
    } else if let Some(account) = store.active_mut()
        && let Some(tokens) = account.tokens.take()
    {
        changed = true;
        confirmed = revoke(&tokens.refresh, &account.client).await;
    }
    if changed {
        save_store(&path, &store)?;
    }
    Ok(confirmed)
}

async fn revoke(refresh_token: &str, client_id: &str) -> bool {
    let mut confirmed = refresh_token.is_empty();
    if !confirmed && let Ok(http) = client() {
        for attempt in 0..2 {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            match http
                .post(REVOKE)
                .form(&[
                    ("token", refresh_token),
                    ("token_type_hint", "refresh_token"),
                    ("client_id", client_id),
                ])
                .send()
                .await
            {
                Ok(response) if response.status() == reqwest::StatusCode::OK => {
                    confirmed = true;
                    break;
                }
                Ok(response) if !response.status().is_server_error() => break,
                _ => {}
            }
        }
    }
    confirmed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorization_uses_public_client_with_pkce_and_stable_redirect() {
        let redirect = "http://127.0.0.1:49152/auth/callback";
        let url = authorize_url(
            "urn:uuid:host",
            None,
            redirect,
            "state",
            "nonce",
            "verifier",
            false,
        )
        .unwrap();
        let p: std::collections::HashMap<_, _> = url.query_pairs().collect();
        assert_eq!(p["client_id"], DYNAMIC);
        assert_eq!(p["agent_name_hint"], "EdgeCrab");
        assert_eq!(p["resource"], RESOURCE);
        assert_eq!(p["redirect_uri"], redirect);
        assert_eq!(p["nonce"], "nonce");
        assert_eq!(p["state"], "state");
        assert_eq!(p["code_challenge"], code_challenge("verifier"));
        assert!(p["scope"].split_whitespace().any(|s| s == PLAN_SCOPE));
        let url = authorize_url(
            "urn:uuid:host",
            Some("oaiapp_first"),
            redirect,
            "state",
            "nonce",
            "verifier",
            true,
        )
        .unwrap();
        let p: std::collections::HashMap<_, _> = url.query_pairs().collect();
        assert_eq!(p["client_id"], "oaiapp_first");
        assert_eq!(p["prompt"], "consent");
        assert!(!p.contains_key("id_token_hint"));
        assert!(!p.contains_key("agent_name_hint"));
    }

    #[test]
    fn callback_requires_issued_client_and_rejects_client_swaps() {
        let mut reply = Callback {
            state: Some("s".into()),
            code: Some("c".into()),
            client_id: None,
            error: None,
        };
        assert!(issued_client(&reply, None).is_err());
        assert_eq!(
            issued_client(&reply, Some("oaiapp_first")).unwrap(),
            "oaiapp_first"
        );
        reply.client_id = Some("oaiapp_second".into());
        assert_eq!(issued_client(&reply, None).unwrap(), "oaiapp_second");
        assert!(issued_client(&reply, Some("oaiapp_first")).is_err());
        reply.error = Some("access_denied".into());
        assert!(issued_client(&reply, None).is_err());
    }

    #[test]
    fn verified_claims_must_match_attempt() {
        let mut claims = Claims {
            iss: ISSUER.into(),
            sub: "user".into(),
            email: None,
            aud: Audience::Single("oaiapp_a".into()),
            azp: None,
            exp: 200,
            nbf: None,
            nonce: "nonce".into(),
        };
        assert!(validate_claims(&claims, "oaiapp_a", "nonce", 100).is_ok());
        assert!(validate_claims(&claims, "oaiapp_b", "nonce", 100).is_err());
        assert!(validate_claims(&claims, "oaiapp_a", "other", 100).is_err());
        assert!(validate_claims(&claims, "oaiapp_a", "nonce", 200).is_err());
        claims.iss = "https://other.invalid".into();
        assert!(validate_claims(&claims, "oaiapp_a", "nonce", 100).is_err());
    }

    #[test]
    fn atomic_store_keeps_registrations_separate_and_reusable() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(FILENAME);
        let mut store = Store {
            host: "host".into(),
            ..Store::default()
        };
        let account = |client: &str, subject: &str| Account {
            client: client.into(),
            subject: subject.into(),
            email: Some("same@example.invalid".into()),
            tokens: None,
        };
        store.replace(account("oaiapp_a", "same-subject")).unwrap();
        store.replace(account("oaiapp_b", "same-subject")).unwrap();
        assert!(store.replace(account("oaiapp_a", "wrong-subject")).is_err());
        save_store(&file, &store).unwrap();
        let reread = read_store(&file).unwrap();
        assert_eq!(reread.host, "host");
        assert_eq!(reread.accounts.len(), 2);
        assert_eq!(reread.active().unwrap().client, "oaiapp_b");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(file).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn refresh_rotates_and_retains_scopes_only_if_omitted() {
        let mut old = Tokens {
            access: "old-access".into(),
            refresh: "old-refresh".into(),
            id: "id".into(),
            scope: SCOPE.into(),
            expires_at: 110,
        };
        assert!(old.needs_refresh(50));
        assert!(!old.needs_refresh(49));
        let response = |scope| TokenResponse {
            access_token: "new-access".into(),
            refresh_token: Some("new-refresh".into()),
            id_token: None,
            token_type: "Bearer".into(),
            expires_in: 3_600,
            scope,
        };
        let rotated = response(None)
            .credentials(Some(&old), old.id.clone())
            .unwrap();
        assert_eq!(rotated.refresh, "new-refresh");
        assert_eq!(rotated.access, "new-access");
        assert!(rotated.allows_plan());
        old = response(Some("openid".into()))
            .credentials(Some(&rotated), rotated.id.clone())
            .unwrap();
        assert!(!old.allows_plan());
        assert!(
            TokenError {
                status: Some(reqwest::StatusCode::BAD_REQUEST),
                code: Some("invalid_grant".into())
            }
            .terminal_refresh()
        );
        assert!(
            !TokenError {
                status: Some(reqwest::StatusCode::SERVICE_UNAVAILABLE),
                code: Some("invalid_grant".into())
            }
            .terminal_refresh()
        );
    }
}
