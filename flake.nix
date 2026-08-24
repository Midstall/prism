{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flakever.url = "github:numinit/flakever";
  };

  outputs =
    {
      self,
      nixpkgs,
      treefmt-nix,
      flakever,
      ...
    }@inputs:
    let
      inherit (nixpkgs) lib;

      nameValuePair = name: value: { inherit name value; };
      genAttrs = names: f: builtins.listToAttrs (map (n: nameValuePair n (f n)) names);
      allSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      flakeverConfig = flakever.lib.mkFlakever {
        inherit inputs;

        digits = [
          1
          2
          2
        ];
      };

      forAllSystems =
        f:
        genAttrs allSystems (
          system:
          f {
            inherit system;
            pkgs = import nixpkgs {
              inherit system;
            };
          }
        );

      treefmtEval = forAllSystems ({ pkgs, ... }: treefmt-nix.lib.evalModule pkgs (import ./treefmt.nix));
    in
    {
      versionTemplate = "1.1pre-<lastModifiedDate>-<rev>";

      devShells = forAllSystems (
        { pkgs, ... }:
        {
          default = pkgs.mkShell {
            packages =
              with pkgs;
              (
                [
                  zig
                ]
                ++ lib.optionals (stdenv.hostPlatform.isLinux) [
                  pkg-config
                  wayland
                ]
              );
          };
        }
      );

      formatter = forAllSystems ({ system, ... }: treefmtEval.${system}.config.build.wrapper);

      checks = forAllSystems (
        { system, pkgs, ... }:
        {
          default = pkgs.stdenv.mkDerivation (finalAttrs: {
            pname = "prism";
            inherit (flakeverConfig) version;

            src = lib.cleanSource ./.;

            zigDeps = pkgs.zig.fetchDeps {
              inherit (finalAttrs) src pname version;
              hash = "sha256-Zcnbxtarbp7VxcMTs8+oz0/XPB3lDkAy+c9/W01GRio=";
            };

            nativeBuildInputs =
              with pkgs;
              [
                zig
              ]
              ++ lib.optional (pkgs.stdenv.hostPlatform.isLinux) pkg-config;

            buildInputs = with pkgs; (lib.optional (pkgs.stdenv.hostPlatform.isLinux) wayland);

            postConfigure = ''
              ln -s ${finalAttrs.zigDeps} "$ZIG_GLOBAL_CACHE_DIR/p"
            '';

            doCheck = true;
          });

          formatting = treefmtEval.${system}.config.build.check self;
        }
      );
    };
}
