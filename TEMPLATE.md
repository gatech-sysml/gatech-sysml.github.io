Lab Website Template: v1.4.0
Upstream: https://github.com/greenelab/lab-website-template
Commit: cca827da53417f039cb485bc8930a6d163aaa48a

Migrated framework files from v1.2.0 using a controlled file migration. SAIL content,
roles, analytics, dark appearance, footer, blog descriptions and publication styling
are preserved. Example content and template-maintenance workflows are excluded.
Upstream multi-author rendering accepts SAIL's existing `authors` metadata.
Docker and CI use Ruby 3.4; the Docker base is pinned to its multi-platform digest.

Verification completed locally:
- Baseline and upgraded inventory: 48 HTML pages, including 41 member pages.
- Production and /preview/pr-123 builds: no missing internal links or assets.
- Chrome at 1440px and 390px: home, research, team, member, blog and both posts;
  mobile navigation and publication search passed, with no page JavaScript errors
  or horizontal overflow. Author links and local analytics suppression checked.
- Team roles retained: PI 1, PhD 6, masters 16, undergraduate 1, and alumni 17.
- Live reload and configuration restart passed; tracked file hashes unchanged.
- Explicit citation generation succeeded; committed citations retained afterward.
- Workflow YAML and actionlint passed. GitHub execution remains to be validated.
- Disposable volume/image cleanup and rebuild/restart passed.
