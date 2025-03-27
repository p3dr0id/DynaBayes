# Makefile

# Default commit message with date (can be overridden)
MSG ?= Update: $(shell date "+%Y-%m-%d %H:%M:%S")

update:
	git add .
	git commit -m "$(MSG)"
	git push origin main
