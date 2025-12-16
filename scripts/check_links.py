"""Link checker for documentation.

Validates all URLs in markdown files and reports broken links.
"""

import re
import sys
import argparse
from pathlib import Path
from urllib.parse import urlparse
import urllib.request
import urllib.error
import ssl
import socket
from dataclasses import dataclass
from typing import Iterator


@dataclass
class LinkResult:
    """Result of checking a link."""
    file: Path
    line: int
    url: str
    status: str
    error: str | None = None


# Known flaky or rate-limited hosts to skip
SKIP_HOSTS = {
    "localhost",
    "127.0.0.1",
    "example.com",
    "example.org",
}

# Known hosts that may rate-limit (check but don't fail build)
WARN_HOSTS = {
    "github.com",
    "pypi.org",
    "arxiv.org",
}

# URL pattern for markdown
URL_PATTERN = re.compile(
    r'\[([^\]]*)\]\(([^)]+)\)|'  # [text](url)
    r'<(https?://[^>]+)>|'       # <url>
    r'(?<!\()(https?://[^\s\)]+)'  # bare url
)


def extract_urls(file_path: Path) -> Iterator[tuple[int, str]]:
    """Extract all URLs from a markdown file with line numbers."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return
    
    for line_num, line in enumerate(content.splitlines(), 1):
        for match in URL_PATTERN.finditer(line):
            # Get URL from whichever group matched
            url = match.group(2) or match.group(3) or match.group(4)
            if url and url.startswith(("http://", "https://")):
                yield line_num, url


def check_url(url: str, timeout: int = 10) -> tuple[str, str | None]:
    """Check if a URL is reachable.
    
    Returns:
        Tuple of (status, error_message)
        status: "ok", "warn", "error", "skip"
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        
        # Skip known problematic hosts
        if host in SKIP_HOSTS:
            return "skip", None
        
        # Create SSL context that works with most sites
        ctx = ssl.create_default_context()
        
        # Try HEAD first (faster), fall back to GET
        request = urllib.request.Request(
            url,
            method="HEAD",
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; LinkChecker/1.0)",
                "Accept": "*/*",
            }
        )
        
        try:
            response = urllib.request.urlopen(request, timeout=timeout, context=ctx)
            status_code = response.getcode()
        except urllib.error.HTTPError as e:
            if e.code == 405:  # Method not allowed, try GET
                request = urllib.request.Request(
                    url,
                    method="GET",
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; LinkChecker/1.0)",
                        "Accept": "*/*",
                    }
                )
                response = urllib.request.urlopen(request, timeout=timeout, context=ctx)
                status_code = response.getcode()
            else:
                raise
        
        if 200 <= status_code < 400:
            if host in WARN_HOSTS:
                return "warn", None
            return "ok", None
        else:
            return "error", f"HTTP {status_code}"
            
    except urllib.error.HTTPError as e:
        if host in WARN_HOSTS:
            return "warn", f"HTTP {e.code}"
        return "error", f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return "error", str(e.reason)
    except socket.timeout:
        return "warn", "Timeout"
    except Exception as e:
        return "error", str(e)[:50]


def find_markdown_files(root: Path) -> Iterator[Path]:
    """Find all markdown files in directory."""
    for pattern in ["*.md", "**/*.md"]:
        yield from root.glob(pattern)


def check_links(root: Path, verbose: bool = False) -> list[LinkResult]:
    """Check all links in markdown files under root."""
    results: list[LinkResult] = []
    checked_urls: dict[str, tuple[str, str | None]] = {}
    
    md_files = sorted(set(find_markdown_files(root)))
    
    for file_path in md_files:
        if verbose:
            print(f"Checking: {file_path.relative_to(root)}")
        
        for line_num, url in extract_urls(file_path):
            # Cache URL checks
            if url not in checked_urls:
                status, error = check_url(url)
                checked_urls[url] = (status, error)
            else:
                status, error = checked_urls[url]
            
            results.append(LinkResult(
                file=file_path.relative_to(root),
                line=line_num,
                url=url,
                status=status,
                error=error,
            ))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Check links in markdown files")
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to check (default: current directory)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    args = parser.parse_args()
    
    root = Path(args.root)
    if not root.exists():
        print(f"Error: {root} does not exist")
        sys.exit(1)
    
    print(f"Checking links in: {root.absolute()}")
    print("-" * 50)
    
    results = check_links(root, verbose=args.verbose)
    
    # Count by status
    counts = {"ok": 0, "warn": 0, "error": 0, "skip": 0}
    for r in results:
        counts[r.status] += 1
    
    # Report errors
    errors = [r for r in results if r.status == "error"]
    warnings = [r for r in results if r.status == "warn"]
    
    if errors:
        print("\nBroken links:")
        for r in errors:
            print(f"  {r.file}:{r.line}")
            print(f"    {r.url}")
            print(f"    Error: {r.error}")
    
    if warnings and args.verbose:
        print("\nWarnings:")
        for r in warnings:
            print(f"  {r.file}:{r.line} - {r.url}")
            if r.error:
                print(f"    {r.error}")
    
    print("-" * 50)
    print(f"Total: {len(results)} links")
    print(f"  OK: {counts['ok']}")
    print(f"  Warnings: {counts['warn']}")
    print(f"  Errors: {counts['error']}")
    print(f"  Skipped: {counts['skip']}")
    
    # Exit code
    if errors:
        sys.exit(1)
    if args.strict and warnings:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
