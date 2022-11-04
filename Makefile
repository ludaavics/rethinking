# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = docs


docs:
	@jupytext --to notebook notebooks/*.md
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)"
	@$(SPHINXBUILD) -anE -b html "$(SOURCEDIR)" "$(BUILDDIR)"
	@touch "$(BUILDDIR)"/.nojekyll

docs-clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)"
	@touch "$(BUILDDIR)"/.nojekyll


.PHONY: docs	help Makefile
