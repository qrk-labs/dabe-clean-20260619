TOOLS_BIN := ../.tools/bin
TECTONIC := $(TOOLS_BIN)/tectonic

.PHONY: install-tex pdf clean

install-tex:
	./install_workspace_tex.sh

pdf:
	PATH="$(abspath $(TOOLS_BIN)):$$PATH" tectonic main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.toc main.xdv
