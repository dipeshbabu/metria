# Build and publish the documentation

The documentation site builds the maintained Markdown in `docs/` with
MkDocs Material. The repository remains the source of truth; no generated
HTML is committed.

## Preview locally

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root:

```bash
uv run --locked --script tools/docs/site.py serve
```

Open the URL printed by MkDocs. The site uses the same `/metria/` base path as
GitHub Pages, including nested pages and assets.

## Validate a change

```bash
uv run --locked --script tools/docs/site.py build --strict
python tools/docs/check_site.py site
```

The strict build rejects missing navigation entries and broken document links.
The HTML check follows local page, asset, and fragment references, and verifies
that the search index is present. Review desktop and narrow layouts, search,
code-copy controls, and both color themes after changing the design.

The site builder has a separate PEP 723 lockfile,
`tools/docs/site.py.lock`. Documentation dependencies do not become runtime
dependencies of the published Metria package. Update them with:

```bash
uv lock --script tools/docs/site.py
uv audit --locked --script tools/docs/site.py
```

## Add or edit a page

1. Edit Markdown under `docs/`.
2. Add new pages to the appropriate section of `mkdocs.yml`.
3. Prefer relative Markdown links for other documentation pages.
4. Keep current release guidance separate from staged and historical evidence.
5. Run the checks above before opening a pull request.

Use commas, colons, semicolons, or parentheses instead of em dashes. Hyphens in
compound words are fine. The documentation build enforces this rule, including
HTML entities that render as an em dash.

Links to files outside `docs/`, such as examples or retained artifacts, remain
ordinary repository-relative links in source. A build hook changes those links
to the matching GitHub revision for the published site. It does not copy research
data, local notes, or private inputs into the site.

The SVG logo and custom styles are in `docs/assets/`. The same transparent
symbol is used in the header and as the site favicon.

## GitHub Pages deployment

The **Documentation** workflow builds and checks pull requests, then deploys
successful builds of `main` through GitHub Actions. Pull requests cannot deploy.

Repository setup uses **Settings → Pages → Build and deployment → GitHub Actions**.
The deployment job uses the `github-pages` environment with `pages: write` and
`id-token: write`; the build job has read-only repository access.

Published address: <https://dipeshbabu.github.io/metria/>.

A manual workflow dispatch on `main` can rebuild the current site. The workflow
uses the checked-in lockfile and runs the link checks before uploading its Pages
artifact. Package releases and documentation deployment are separate workflows.
