from pathlib import Path

def make_summary(data):
    """Return a dummy summary string for now."""
    return "Summary: dummy data"

def save_text_report(text, path):
    """Save the summary text to a file at the given path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved report to {path}")