# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in AgentBible, please report it responsibly:

1. **Do NOT open a public GitHub issue** for security vulnerabilities
2. **Email**: Send details to [rylan1012@gmail.com](mailto:rylan1012@gmail.com)
3. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 1-2 weeks
  - Medium: 1 month
  - Low: Next release

## Security Measures

### Dependency Scanning

We use `pip-audit` in CI to scan for known vulnerabilities in dependencies. The CI pipeline fails on any detected vulnerability.

### Code Security

- No arbitrary code execution in templates
- No secrets stored in repository
- Pre-commit hooks detect potential credential leaks

### Trusted Publishing

PyPI releases use GitHub Actions trusted publishing (OIDC) - no stored credentials that could be compromised.

## Best Practices for Users

1. **Keep updated**: Install the latest version
2. **Review templates**: Before using, review template contents
3. **Don't commit secrets**: Use `.env` files and `.gitignore`
4. **Report issues**: Help us improve security

## Scope

This security policy covers:
- The `agentbible` Python package
- CLI tools (`bible` command)
- Project templates
- GitHub Actions workflows

It does NOT cover:
- Third-party dependencies (report to their maintainers)
- Projects generated from templates (your responsibility)
- Self-hosted infrastructure

## Acknowledgments

We appreciate responsible disclosure and will acknowledge reporters in release notes (with permission).
