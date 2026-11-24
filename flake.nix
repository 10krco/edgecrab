{
  description = "EdgeCrab development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { nixpkgs, rust-overlay, ... }:
    {
      devShells =
        nixpkgs.lib.genAttrs
          [
            "x86_64-linux"
            "aarch64-linux"
            "aarch64-darwin"
          ]
          (
            system:
            let
              pkgs = import nixpkgs {
                inherit system;
                overlays = [ rust-overlay.overlays.default ];
              };
              rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
            in
            {
              default = pkgs.mkShell {
                packages = with pkgs; [
                  rust
                  git
                  gnumake
                  pkg-config

                  # Used by test helpers and local tool discovery.
                  curl
                  which
                  ripgrep

                  # Native crypto builds (BoringSSL and AWS-LC) and bindgen.
                  cmake
                  ninja
                  go
                  clang
                  rustPlatform.bindgenHook

                  # Workspace SDKs and documentation site.
                  python312
                  maturin
                  nodejs_24
                  pnpm_10
                  wasm-pack
                ];
              };
            }
          );
    };
}
