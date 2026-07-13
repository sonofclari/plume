GIT_ENV = GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.bareRepository GIT_CONFIG_VALUE_0=all

build:
	$(GIT_ENV) swift build

release:
	$(GIT_ENV) swift build -c release

run:
	$(GIT_ENV) swift run App

clean:
	swift package clean

.PHONY: build release run clean
