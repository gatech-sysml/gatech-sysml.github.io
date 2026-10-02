# Webpage for Systems for AI Lab at Georgia Tech

Visit **[gatech-sysml.github.io](https://gatech-sysml.github.io)** 🚀

Built with [Lab Website Template](https://greene-lab.gitbook.io/lab-website-template-docs).
Template version and migration details are recorded in [TEMPLATE.md](TEMPLATE.md).

## Local development with Docker

Start your Docker engine, then run from this repository:

```sh
docker compose up --build site
```

Open http://localhost:4000. Source edits trigger live reload; changes to
`_config.yaml` restart Jekyll (refresh your browser afterward).
`./.docker/run.sh` runs the same preview command. Ports bind only to localhost.
Local previews use committed citations and disable analytics. Production URL
settings in `_config.yaml` remain intact.

Build the production site or explicitly refresh citations:

```sh
docker compose run --rm build
docker compose run --rm cite
```

Citation generation may use external APIs and updates `_data/citations.yaml`;
review that diff before committing. Set `GOOGLE_SCHOLAR_API_KEY` with
`docker compose run --rm -e GOOGLE_SCHOLAR_API_KEY cite` if needed.
Scheduled GitHub citation updates remain enabled.

Check a nested PR preview path with:

```sh
docker compose run --rm build build --baseurl /preview/pr-123
```

The generated site lives in the `site-output` Docker volume. To copy it out:

```sh
docker compose run --rm -v "$PWD/site-export:/export" build cp -a _site/. /export
```

The image contains Ruby and a Python virtual environment, so neither language
needs to be installed locally. Both ARM64 and AMD64 Linux dependencies are in
the lockfile. Rebuild the image after changing dependency files.

Stop the preview with Ctrl-C. Remove this project's containers, disposable build
and cache volumes, and locally built image with:

```sh
docker compose down --volumes --rmi local
```

Source edits and deliberately generated citations remain in the repository.
GitHub Pages continues to publish from `gh-pages`. The upgraded GitHub PR build
has passed; production deployment and the new preview deployment workflow still
need verification after merge.
