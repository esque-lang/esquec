{
  description = "esquec: the esque language compiler";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      pkgsFor = system: nixpkgs.legacyPackages.${system};

      version =
        if self ? shortRev then self.shortRev
        else if self ? dirtyShortRev then self.dirtyShortRev
        else "0.13-dev";
    in
    {
      packages = forAllSystems (system:
        let pkgs = pkgsFor system; in
        rec {
          esquec = pkgs.buildGoModule {
            pname = "esquec";
            inherit version;
            src = ./.;
            vendorHash = "sha256-zckNINzay7prNC8bGLfYLDHeG6HVF+3Ft/Qe7iuP9Y8=";
            subPackages = [ "cmd/esquec" ];
            ldflags = [ "-s" "-w" ];
            meta = with pkgs.lib; {
              description = "Compiler for the esque programming language";
              homepage = "https://github.com/esque-lang/esquec";
              license = licenses.mit;
              mainProgram = "esquec";
              platforms = platforms.linux;
            };
          };
          default = esquec;
        });

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.esquec}/bin/esquec";
        };
        esquec = self.apps.${system}.default;
      });

      devShells = forAllSystems (system:
        let pkgs = pkgsFor system; in
        {
          default = pkgs.mkShell {
            packages = [ pkgs.go ];
            inputsFrom = [ self.packages.${system}.esquec ];
          };
        });

      overlays.default = final: prev: {
        esquec = self.packages.${final.system}.esquec;
      };

      formatter = forAllSystems (system: (pkgsFor system).nixpkgs-fmt);
    };
}
