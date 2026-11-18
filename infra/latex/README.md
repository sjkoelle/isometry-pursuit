# Automatic manuscript builds

The macOS LaunchAgent runs `latexmk -pdf -pvc -view=none` in `archival/`.
It watches the manuscript's dependencies, including included TeX, bibliography,
style, and figure files, and updates `archival/main.pdf` after changes.
It starts at login and does not open or activate a viewer.
Logs are `archival/automatic-build.log` and `archival/automatic-build-error.log`.
TeXShop controls whether its open preview reloads the updated PDF.

Install and start from the repository root:

```sh
cp infra/latex/org.isometry-pursuit.manuscript-watch.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/org.isometry-pursuit.manuscript-watch.plist
```

Stop with:

```sh
launchctl bootout gui/$(id -u)/org.isometry-pursuit.manuscript-watch
```

To disable startup at login, also remove the installed plist from
`~/Library/LaunchAgents/`.
The plist contains this checkout's absolute path; update it if the repository moves.
Avoid running a second build concurrently with the watcher.
