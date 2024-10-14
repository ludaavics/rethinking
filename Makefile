# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = docs

notebooks:
	@jupytext --update --to notebook notebooks/*.md

docs:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)"
	@$(SPHINXBUILD) -nE -j auto -b html "$(SOURCEDIR)" "$(BUILDDIR)"
	@touch "$(BUILDDIR)"/.nojekyll

docs-clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)"
	@touch "$(BUILDDIR)"/.nojekyll


.PHONY: docs	help Makefile
