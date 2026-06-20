TOOLS_BIN := ../.tools/bin

.PHONY: install-typst pdf clean

install-typst:
	./install_workspace_typst.sh

pdf:
	PATH="$(abspath $(TOOLS_BIN)):$$PATH" typst compile --root .. main.typ main.pdf

clean:
	rm -f main.pdf
