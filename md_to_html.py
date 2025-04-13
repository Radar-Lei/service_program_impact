#!/usr/bin/env python
import os
import re
import markdown
from pathlib import Path

def convert_md_to_html(md_file="service_program_impact_report.md", html_file="service_program_impact_report.html"):
    """
    Convert Markdown file to HTML with styling
    
    Args:
        md_file (str): Path to the markdown file
        html_file (str): Path to output HTML file
    
    Returns:
        bool: True if conversion was successful
    """
    current_dir = Path.cwd()
    
    # Read the Markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Process images to ensure correct paths
    def process_images_in_markdown(content):
        # Function to replace image markdown with correct relative paths
        def replace_image(match):
            alt_text = match.group(1)
            image_path = match.group(2)
            return f"![{alt_text}]({image_path})"
        
        # Replace all image references
        content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image, content)
        return content

    # Pre-process markdown content
    md_content = process_images_in_markdown(md_content)

    # Convert Markdown to HTML with extensions
    html_content = markdown.markdown(
        md_content,
        extensions=[
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.codehilite',
            'markdown.extensions.extra',
            'markdown.extensions.toc',
        ]
    )

    # Add proper CSS for styling
    css_content = """
/* Font settings */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700&display=swap');

body {
    font-family: 'Noto Sans SC', 'Arial', sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1000px;
    margin: 0 auto;
    padding: 20px;
    font-size: 12pt;
}

h1, h2, h3, h4, h5, h6 {
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    color: #2c3e50;
}

h1 {
    font-size: 24pt;
    border-bottom: 2px solid #eaecef;
    padding-bottom: 0.3em;
    text-align: center;
}

h2 {
    font-size: 20pt;
    border-bottom: 1px solid #eaecef;
    padding-bottom: 0.3em;
}

h3 {
    font-size: 16pt;
}

p {
    margin-top: 0.5em;
    margin-bottom: 1em;
    text-align: justify;
}

img {
    max-width: 100%;
    height: auto;
    margin: 20px auto;
    display: block;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    font-size: 10pt;
}

th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}

th {
    background-color: #f2f2f2;
    font-weight: bold;
}

code {
    background-color: #f5f5f5;
    border-radius: 3px;
    padding: 2px 4px;
    font-family: monospace;
    font-size: 10pt;
}

pre {
    background-color: #f5f5f5;
    border-radius: 3px;
    padding: 15px;
    overflow-x: auto;
    margin-bottom: 20px;
}

pre code {
    background-color: transparent;
    padding: 0;
}

blockquote {
    border-left: 4px solid #ddd;
    padding-left: 16px;
    margin-left: 0;
    color: #555;
}

/* LaTeX styling with MathJax */
.MathJax {
    display: inline-block !important;
}
"""

    # Add MathJax support for rendering LaTeX expressions
    mathjax_script = """
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
      displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
      processEscapes: true,
      tags: "ams"
    },
    options: {
      ignoreHtmlClass: "no-mathjax",
      processHtmlClass: "mathjax"
    }
  };
</script>
"""

    # Create complete HTML file
    full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Service Program Impact Report</title>
    <style>
    {css_content}
    </style>
    {mathjax_script}
</head>
<body>
    {html_content}
    <script>
    // Add a small script to fix image paths if needed
    document.addEventListener('DOMContentLoaded', function() {{
        const images = document.querySelectorAll('img');
        images.forEach(img => {{
            if (!img.src.startsWith('http') && !img.src.startsWith('file://')) {{
                // Convert relative paths
                img.src = img.src;
            }}
        }});
    }});
    </script>
</body>
</html>
"""

    # Save complete HTML
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f"HTML file generated at {html_file}")
    print("\nTo convert to PDF:")
    print("1. Open the HTML file in a browser: Chrome or Safari")
    print("2. Use Print (Cmd+P or Ctrl+P)")
    print("3. Choose 'Save as PDF' as the destination")
    print("4. Click 'Save' to create your PDF file")
    
    return True

# If run as a script (not imported)
if __name__ == "__main__":
    convert_md_to_html()
