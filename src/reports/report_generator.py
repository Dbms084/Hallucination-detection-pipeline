import os
import sys
import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.scoring.disagreement_analytics import calculate_disagreement_stats

class NumberedCanvas(canvas.Canvas):
    """
    Canvas to dynamically compute and display total page count in the footer,
    and draw consistent headers and footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Header (Skip on first page)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#475569"))
            self.drawString(54, 755, "HALLUCINATION DETECTION & LLM EVALUATION PLATFORM")
            self.setFont("Helvetica", 8)
            self.drawRightString(558, 755, "Evaluation Summary Report")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 747, 558, 747)

        # Footer (On all pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 50, 558, 50)
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, 38, "Confidential - Generated Automatically")
        
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 38, page_text)
        
        self.restoreState()


def build_pdf_report(csv_path, output_pdf_path, threshold=0.6, vis_dir="results/visualizations"):
    """
    Generate a formatted PDF report summarizing the evaluation run.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    total_q = len(df)
    
    if total_q == 0:
        print("Error: Evaluation results CSV is empty.")
        return False
        
    # Create reports directory if it doesn't exist
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

    # 1. Base Metrics Setup
    exact_acc = (df["exact_match"].sum() / total_q) * 100
    keyword_acc = (df["keyword_match"].sum() / total_q) * 100
    semantic_acc = ((df["semantic_score"] >= threshold).sum() / total_q) * 100
    avg_semantic_score = df["semantic_score"].mean()
    
    has_judge = "judge_score" in df.columns
    judge_valid_df = df.dropna(subset=["judge_score"]) if has_judge else pd.DataFrame()
    judge_evaluated_count = len(judge_valid_df)
    
    if has_judge and judge_evaluated_count > 0:
        judge_acc = (judge_valid_df["judge_score"].sum() / judge_evaluated_count) * 100
    else:
        judge_acc = None

    # Calculate disagreement stats
    disagreement_stats = calculate_disagreement_stats(df, threshold)

    # 2. Document Template Setup
    # Letter size: 612 x 792 pt
    # Margins: 0.75 in (54 pt) left/right, 1 in (72 pt) top/bottom
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    styles = getSampleStyleSheet()

    # Define custom styling
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0F172A"),
        alignment=0,
        spaceAfter=6
    )

    h1_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        "SubSectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0D9488"),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "BodyText_Custom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )

    body_bold = ParagraphStyle(
        "BodyText_Bold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    meta_style = ParagraphStyle(
        "MetadataText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        "TableHeaderText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        "TableCellText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#1E293B"),
        alignment=0
    )

    table_cell_center = ParagraphStyle(
        "TableCellTextCenter",
        parent=table_cell_style,
        alignment=1
    )

    table_cell_bold = ParagraphStyle(
        "TableCellTextBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold"
    )

    code_cell_style = ParagraphStyle(
        "CodeCellText",
        parent=table_cell_style,
        fontName="Courier",
        fontSize=7.5,
        leading=9
    )

    story = []

    # --- PAGE 1: TITLE & EXECUTIVE METRICS ---
    
    # Title Block
    story.append(Paragraph("Evaluation Summary Report", title_style))
    story.append(Paragraph("<b>LLM Hallucination Detection & Performance Benchmarks</b>", ParagraphStyle("Subtitle", parent=body_style, fontSize=11, leading=15, textColor=colors.HexColor("#475569"))))
    story.append(Spacer(1, 10))

    # Metadata Row (Format nicely using a grid)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    meta_text = f"<b>Generated:</b> {current_time} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Dataset:</b> TruthfulQA (Generation) &nbsp;&nbsp;|&nbsp;&nbsp; <b>Total Evaluated:</b> {total_q} questions"
    story.append(Paragraph(meta_text, meta_style))
    
    # Horizontal divider
    divider = Table([[""]], colWidths=[504], rowHeights=[2])
    divider.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#1E3A8A")),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(divider)
    story.append(Spacer(1, 12))

    # Executive Summary Paragraph
    summary_p = (
        "This automated report evaluates the factual correctness and semantic accuracy of LLM responses against "
        "curated ground-truth targets. By combining rule-based lexical matching (Exact & Keyword Match), "
        "vector embeddings similarity (Semantic Similarity), and LLM-as-a-Judge evaluations, the pipeline isolates "
        "cases of term-overlap hallucinations, paraphrased correctness, and outright factual failures."
    )
    story.append(Paragraph(summary_p, body_style))
    story.append(Spacer(1, 10))

    # Executive Summary Metrics Table
    story.append(Paragraph("Executive Performance Metrics", h1_style))
    
    headers = [
        Paragraph("Metric", table_header_style),
        Paragraph("Description", table_header_style),
        Paragraph("Accuracy / Value", table_header_style)
    ]
    
    metrics_data = [
        headers,
        [
            Paragraph("<b>Exact Match</b>", table_cell_style),
            Paragraph("Literal, string-by-string equality with the target answer.", table_cell_style),
            Paragraph(f"<b>{exact_acc:.2f}%</b>", table_cell_center)
        ],
        [
            Paragraph("<b>Keyword Match</b>", table_cell_style),
            Paragraph("Presence of essential target key-terms in model response.", table_cell_style),
            Paragraph(f"<b>{keyword_acc:.2f}%</b>", table_cell_center)
        ],
        [
            Paragraph("<b>Semantic Match</b>", table_cell_style),
            Paragraph(f"Cosine similarity of sentence embeddings &ge; {threshold:.2f}.", table_cell_style),
            Paragraph(f"<b>{semantic_acc:.2f}%</b>", table_cell_center)
        ],
        [
            Paragraph("<b>Average Cos Sim</b>", table_cell_style),
            Paragraph("Mean cosine similarity score across all evaluated pairs.", table_cell_style),
            Paragraph(f"<b>{avg_semantic_score:.3f}</b>", table_cell_center)
        ]
    ]
    
    if has_judge:
        judge_status = f"<b>{judge_acc:.2f}%</b>" if judge_acc is not None else "N/A"
        metrics_data.append([
            Paragraph("<b>Gemini Judge</b>", table_cell_style),
            Paragraph("Factual verification by an independent LLM Judge (Gemini).", table_cell_style),
            Paragraph(judge_status, table_cell_center)
        ])
        
    metrics_table = Table(metrics_data, colWidths=[110, 294, 100])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 15))

    # --- Embedded Charts ---
    story.append(Paragraph("Evaluator Accuracy & Threshold Analysis Charts", h1_style))
    
    chart_files = ["evaluator_comparison.png", "threshold_analysis.png"]
    chart_cells = []
    
    for f in chart_files:
        path = os.path.join(vis_dir, f)
        if os.path.exists(path):
            # Letter print width is 504. Two charts side-by-side: 240 pt width each
            img = Image(path, width=242, height=151) # maintain 8:5 ratio
            chart_cells.append(img)
            
    if len(chart_cells) == 2:
        chart_table = Table([[chart_cells[0], chart_cells[1]]], colWidths=[252, 252])
        chart_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(chart_table)
    elif len(chart_cells) == 1:
        story.append(chart_cells[0])
    else:
        story.append(Paragraph("<i>Charts not generated. Run evaluation scorer.py first to construct visualization assets.</i>", ParagraphStyle("Italic", parent=body_style)))
        
    story.append(PageBreak())

    # --- PAGE 2: DISAGREEMENT & CATEGORY ANALYTICS ---
    
    # 1. Disagreement Analytics Section
    if disagreement_stats:
        story.append(Paragraph("⚖️ Evaluator Disagreement Analytics", h1_style))
        story.append(Paragraph(
            "Disagreements between automated metrics and LLM Judge verdicts highlight structural flaws in evaluations "
            "like literal match failures (paraphrases) or blind term overlap (hallucinations with keywords).",
            body_style
        ))
        story.append(Spacer(1, 5))

        # Horizontal layout: Left table, right donut chart
        dis_table_data = [
            [Paragraph("<b>Disagreement Type</b>", table_header_style), Paragraph("<b>Count</b>", table_header_style), Paragraph("<b>Rate</b>", table_header_style)],
            [
                Paragraph("<b>False Positives</b> (Semantic Pass, Judge Fail)", table_cell_style),
                Paragraph(str(disagreement_stats["false_positive_count"]), table_cell_center),
                Paragraph(f"{disagreement_stats['false_positive_rate']:.1f}%", table_cell_center)
            ],
            [
                Paragraph("<b>False Negatives</b> (Semantic Fail, Judge Pass)", table_cell_style),
                Paragraph(str(disagreement_stats["false_negative_count"]), table_cell_center),
                Paragraph(f"{disagreement_stats['false_negative_rate']:.1f}%", table_cell_center)
            ],
            [
                Paragraph("<b>Keyword Fallacies</b> (Keyword Pass, Judge Fail)", table_cell_style),
                Paragraph(str(disagreement_stats["keyword_fallacy_count"]), table_cell_center),
                Paragraph(f"{disagreement_stats['keyword_fallacy_count']/disagreement_stats['total_evaluated']*100:.1f}%", table_cell_center)
            ],
            [
                Paragraph("<b>Keyword Omissions</b> (Keyword Fail, Judge Pass)", table_cell_style),
                Paragraph(str(disagreement_stats["keyword_omission_count"]), table_cell_center),
                Paragraph(f"{disagreement_stats['keyword_omission_count']/disagreement_stats['total_evaluated']*100:.1f}%", table_cell_center)
            ],
            [
                Paragraph("<b>Overall Disagreement Rate</b>", table_cell_bold),
                Paragraph(str(disagreement_stats["total_disagreed_count"]), table_cell_center),
                Paragraph(f"<b>{disagreement_stats['disagreement_rate']:.1f}%</b>", table_cell_center)
            ]
        ]
        
        dis_table = Table(dis_table_data, colWidths=[200, 45, 55])
        dis_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0D9488")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, colors.HexColor("#F8FAFC")]),
            ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#F1F5F9")),
        ]))
        
        # Load donut chart if exists
        donut_path = os.path.join(vis_dir, "disagreement_distribution.png")
        if os.path.exists(donut_path):
            donut_img = Image(donut_path, width=190, height=190)
            # Create a layout table for Table + Image
            layout_data = [[dis_table, donut_img]]
            layout_table = Table(layout_data, colWidths=[310, 194])
            layout_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('LEFTPADDING', (0,0), (-1,-1), 0),
                ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ]))
            story.append(layout_table)
        else:
            story.append(dis_table)
            
        story.append(Spacer(1, 10))

    # 2. Category Analysis Section
    story.append(Paragraph("📁 Category Performance & Breakdowns", h1_style))
    category_groups = df.groupby("category")
    
    cat_headers = [
        Paragraph("Category", table_header_style),
        Paragraph("Questions", table_header_style),
        Paragraph("Exact Match", table_header_style),
        Paragraph("Keyword Match", table_header_style),
        Paragraph("Semantic Match", table_header_style),
        Paragraph("Avg Cos Sim", table_header_style)
    ]
    if has_judge:
        cat_headers.append(Paragraph("Judge Acc", table_header_style))
        
    cat_table_data = [cat_headers]
    
    for category, grp in sorted(category_groups):
        c_total = len(grp)
        c_exact = (grp["exact_match"].sum() / c_total) * 100
        c_keyword = (grp["keyword_match"].sum() / c_total) * 100
        c_semantic = ((grp["semantic_score"] >= threshold).sum() / c_total) * 100
        c_avg_sim = grp["semantic_score"].mean()
        
        row = [
            Paragraph(f"<b>{category}</b>", table_cell_style),
            Paragraph(str(c_total), table_cell_center),
            Paragraph(f"{c_exact:.1f}%", table_cell_center),
            Paragraph(f"{c_keyword:.1f}%", table_cell_center),
            Paragraph(f"{c_semantic:.1f}%", table_cell_center),
            Paragraph(f"{c_avg_sim:.3f}", table_cell_center)
        ]
        if has_judge:
            c_judge_df = grp.dropna(subset=["judge_score"])
            if len(c_judge_df) > 0:
                c_judge = (c_judge_df["judge_score"].sum() / len(c_judge_df)) * 100
                row.append(Paragraph(f"{c_judge:.1f}%", table_cell_center))
            else:
                row.append(Paragraph("N/A", table_cell_center))
                
        cat_table_data.append(row)
        
    col_widths = [120, 55, 65, 65, 65, 65]
    if has_judge:
        col_widths = [100, 50, 58, 58, 58, 60, 60]
        
    cat_table = Table(cat_table_data, colWidths=col_widths)
    cat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(cat_table)
    
    # Check if there are disagreement cases to list
    disagree_df = pd.DataFrame()
    if has_judge:
        disagree_df = df.dropna(subset=["judge_score", "judge_reason"]).copy()
        disagree_df["semantic_match_dynamic"] = disagree_df["semantic_score"] >= threshold
        disagree_df = disagree_df[disagree_df["semantic_match_dynamic"] != disagree_df["judge_score"]]
        
    if len(disagree_df) > 0:
        story.append(PageBreak())
        
        # --- PAGE 3: DETAILED DISAGREEMENT CASES ---
        story.append(Paragraph("🔍 Selected Evaluator Disagreement Cases", h1_style))
        story.append(Paragraph(
            "Below are specific samples where Semantic Similarity and the Gemini Judge disagreed. "
            "These samples are critical for audit trails to fine-tune evaluation thresholds or identify model hallucinations.",
            body_style
        ))
        story.append(Spacer(1, 10))
        
        # Display up to 5 cases to keep page count under control
        sample_cases = disagree_df.head(5)
        for idx, (_, row) in enumerate(sample_cases.iterrows()):
            case_title = f"Case {idx + 1}: Disagreement in '{row['category']}'"
            story.append(Paragraph(case_title, h2_style))
            
            case_data = [
                [Paragraph("<b>Question:</b>", table_cell_bold), Paragraph(str(row["question"]).replace("\\n", "\n"), table_cell_style)],
                [Paragraph("<b>Ground Truth:</b>", table_cell_bold), Paragraph(str(row["ground_truth"]).replace("\\n", "\n"), table_cell_style)],
                [Paragraph("<b>LLM Response:</b>", table_cell_bold), Paragraph(str(row["response"]).replace("\\n", "\n"), table_cell_style)],
                [
                    Paragraph("<b>Evaluation:</b>", table_cell_bold), 
                    Paragraph(
                        f"Cosine Sim: <b>{row['semantic_score']:.4f}</b> (Match: {'Yes' if row['semantic_match_dynamic'] else 'No'}) &nbsp;&nbsp;|&nbsp;&nbsp; "
                        f"Gemini Judge Score: <b>{int(row['judge_score'])}</b>", 
                        table_cell_style
                    )
                ],
                [Paragraph("<b>Judge Explanation:</b>", table_cell_bold), Paragraph(f"<i>\"{row['judge_reason']}\"</i>", table_cell_style)]
            ]
            
            case_table = Table(case_data, colWidths=[90, 414])
            case_table.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#F8FAFC")),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(case_table)
            story.append(Spacer(1, 10))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF Report built successfully: {output_pdf_path}")
    return True


if __name__ == "__main__":
    # Test CLI execution
    csv_file = "results/csv/evaluation_results.csv"
    pdf_file = "results/reports/evaluation_summary.pdf"
    
    if os.path.exists(csv_file):
        build_pdf_report(csv_file, pdf_file)
    else:
        print(f"Error: Run evaluation first to create {csv_file}")
