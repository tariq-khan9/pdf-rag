import re
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from config import Config


def create_enhanced_pdf_summary(content, filename):
    """Create a well-formatted PDF file with summary content."""
    output_path = os.path.join(Config.DOWNLOAD_FOLDER, f"summary_{filename}")
    
    doc = SimpleDocTemplate(
        output_path, 
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Get default styles and create custom ones
    styles = getSampleStyleSheet()
    
    # Custom styles for better formatting
    custom_styles = {
        'CustomTitle': ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.purple,
            fontName='Helvetica-Bold'
        ),
        'CustomHeading': ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.purple,
            fontName='Helvetica-Bold'
        ),
        'CustomSubheading': ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading3'],
            fontSize=12,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        ),
        'CustomNormal': ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceBefore=6,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ),
        'CustomBullet': ParagraphStyle(
            'CustomBullet',
            parent=styles['Normal'],
            fontSize=11,
            spaceBefore=3,
            spaceAfter=3,
            leftIndent=20,
            bulletIndent=10,
            fontName='Helvetica'
        ),
        'CustomNumbered': ParagraphStyle(
            'CustomNumbered',
            parent=styles['Normal'],
            fontSize=11,
            spaceBefore=3,
            spaceAfter=3,
            leftIndent=20,
            fontName='Helvetica'
        )
    }
    
    story = []
    
    # Title
    title = Paragraph(f"Summary of {filename.replace('_', ' ').replace('.pdf', '')}", custom_styles['CustomTitle'])
    story.append(title)
    story.append(Spacer(1, 20))
    
    # Process the content
    processed_content = format_content_for_pdf(content, custom_styles)
    story.extend(processed_content)
    
    doc.build(story)
    return output_path


def format_content_for_pdf(content, styles):
    """Process content and return formatted story elements."""
    story = []
    
    # Split content into lines and process each one
    lines = content.split('\n')
    current_list_type = None
    list_counter = 0
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines but add small spacer
        if not line:
            if story and not isinstance(story[-1], Spacer):
                story.append(Spacer(1, 6))
            continue
        
        # Check for different formatting patterns
        if is_main_heading(line):
            # Reset list counter for new sections
            current_list_type = None
            list_counter = 0
            
            heading_text = clean_heading_text(line)
            para = Paragraph(heading_text, styles['CustomHeading'])
            story.append(para)
            
        elif is_subheading(line):
            current_list_type = None
            list_counter = 0
            
            subheading_text = clean_heading_text(line)
            para = Paragraph(subheading_text, styles['CustomSubheading'])
            story.append(para)
            
        elif is_bullet_point(line):
            current_list_type = 'bullet'
            bullet_text = clean_bullet_text(line)
            para = Paragraph(f"• {bullet_text}", styles['CustomBullet'])
            story.append(para)
            
        elif is_numbered_point(line):
            if current_list_type != 'numbered':
                current_list_type = 'numbered'
                list_counter = 0
            
            list_counter += 1
            numbered_text = clean_numbered_text(line)
            para = Paragraph(f"{list_counter}. {numbered_text}", styles['CustomNumbered'])
            story.append(para)
            
        else:
            # Regular paragraph
            current_list_type = None
            list_counter = 0
            
            # Handle bold and italic text
            formatted_text = format_inline_text(line)
            para = Paragraph(formatted_text, styles['CustomNormal'])
            story.append(para)
    
    return story


def is_main_heading(line):
    """Check if line is a main heading."""
    heading_patterns = [
        r'^#+\s+',  # Markdown headings
        r'^[A-Z][A-Z\s]{3,}:?\s*$',  # ALL CAPS headings
        r'^\d+\.\s+[A-Z]',  # Numbered headings like "1. Introduction"
        r'^[A-Z][a-z\s]+:$',  # Title case with colon
    ]
    
    for pattern in heading_patterns:
        if re.match(pattern, line):
            return True
    
    # Check if it's a short line (likely heading) with title case
    if len(line) < 50 and len(line.split()) <= 6 and line[0].isupper():
        return True
    
    return False


def is_subheading(line):
    """Check if line is a subheading."""
    subheading_patterns = [
        r'^#{2,}\s+',  # Markdown subheadings
        r'^\d+\.\d+\s+',  # Numbered subheadings like "1.1 Overview"
        r'^[A-Z][a-z]+\s+[A-Z][a-z]+:$',  # Two-word titles with colon
    ]
    
    for pattern in subheading_patterns:
        if re.match(pattern, line):
            return True
    
    return False


def is_bullet_point(line):
    """Check if line is a bullet point."""
    bullet_patterns = [
        r'^[-•*]\s+',  # Dash, bullet, or asterisk
        r'^\s*[-•*]\s+',  # With leading whitespace
        r'^○\s+',  # Circle bullet
        r'^►\s+',  # Arrow bullet
    ]
    
    for pattern in bullet_patterns:
        if re.match(pattern, line):
            return True
    
    return False


def is_numbered_point(line):
    """Check if line is a numbered list item."""
    numbered_patterns = [
        r'^\d+\.\s+',  # 1. Item
        r'^\d+\)\s+',  # 1) Item
        r'^\(\d+\)\s+',  # (1) Item
    ]
    
    for pattern in numbered_patterns:
        if re.match(pattern, line):
            return True
    
    return False


def clean_heading_text(line):
    """Clean heading text by removing formatting markers."""
    # Remove markdown headers
    line = re.sub(r'^#+\s*', '', line)
    # Remove trailing colons
    line = re.sub(r':$', '', line)
    # Clean up extra whitespace
    line = re.sub(r'\s+', ' ', line).strip()
    return line


def clean_bullet_text(line):
    """Clean bullet point text."""
    # Remove bullet markers
    line = re.sub(r'^[-•*○►]\s*', '', line)
    line = re.sub(r'^\s*[-•*○►]\s*', '', line)
    return line.strip()


def clean_numbered_text(line):
    """Clean numbered list text."""
    # Remove numbering
    line = re.sub(r'^\d+\.\s*', '', line)
    line = re.sub(r'^\d+\)\s*', '', line)
    line = re.sub(r'^\(\d+\)\s*', '', line)
    return line.strip()


def format_inline_text(text):
    """Format inline text for bold, italic, etc."""
    # Handle **bold** text
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Handle *italic* text
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Handle __bold__ text
    text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
    # Handle _italic_ text (but not if it's part of filename)
    text = re.sub(r'(?<![\w])_(.*?)_(?![\w])', r'<i>\1</i>', text)
    
    return text