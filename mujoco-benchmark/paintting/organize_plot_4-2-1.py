import os
from PyPDF2 import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from pdf2image import convert_from_path

def get_pdf_page_size(pdf_file):
    """获取PDF文件第一页的宽度和高度。"""
    reader = PdfReader(pdf_file)
    page = reader.pages[0]
    width = float(page.mediabox.width)
    height = float(page.mediabox.height)
    return width, height

def merge_pdfs_with_labels_dynamic(pdf_files, labels, output_path, rows=2, cols=3, dpi=300, label_font_size=14, spacing=20):
    """将多张PDF图片合并为一张PDF，自动根据图片尺寸调整合并后的PDF页面大小，并在每张图片下方添加标签。"""
    
    # 获取所有PDF文件的尺寸（假设所有PDF文件的尺寸相同）
    pdf_width, pdf_height = get_pdf_page_size(pdf_files[0])
    
    # 计算合并后的PDF页面尺寸
    total_width = pdf_width * cols
    total_height = (pdf_height + spacing) * rows  # spacing 用于标签的空间
    
    # 创建一个新的PDF文件，页面大小为动态计算的大小
    c = canvas.Canvas(output_path, pagesize=(total_width, total_height))
    
    # 计算每个子图的位置和大小
    num_pdfs = len(pdf_files)
    for i, (pdf_file, label) in enumerate(zip(pdf_files, labels)):
        # 计算该图在页面中的位置
        col = i % cols
        row = i // cols
        x_position = col * pdf_width
        y_position = total_height - (row + 1) * (pdf_height + spacing)
        
        # 将PDF页面转换为图像
        images = convert_from_path(pdf_file, first_page=0, last_page=1, dpi=dpi)
        img = images[0]
        
        # 绘制图像在 PDF 页面上
        c.drawInlineImage(img, x_position, y_position + spacing, width=pdf_width, height=pdf_height)
        
        # 在图像下方添加标签，标签居中
        c.setFont("Helvetica-Bold", label_font_size)
        label_width = c.stringWidth(label, "Helvetica-Bold", label_font_size)  # 调整为和标签字体相同的大小
        label_x_position = x_position + (pdf_width - label_width) / 2  # 居中计算
        c.drawString(label_x_position, y_position + 5, label)
        
        # 如果已经填满一页，则新建一页
        if (i + 1) % (rows * cols) == 0:
            c.showPage()
    
    # 保存最终的合并PDF
    c.save()


# 使用示例
pdf_files = [ 
    "mujoco-benchmark/paintting/humanoidbench-paintting/plot_4-2/plot_4-2-1/h1hand-cube-v0/h1hand-cube-v0.pdf", 
    "mujoco-benchmark/paintting/humanoidbench-paintting/plot_4-2/plot_4-2-1/h1hand-powerlift-v0/h1hand-powerlift-v0.pdf",
    "mujoco-benchmark/paintting/humanoidbench-paintting/plot_4-2/plot_4-2-1/h1hand-bookshelf_simple-v0/h1hand-bookshelf_simple-v0.pdf",
    "mujoco-benchmark/paintting/humanoidbench-paintting/plot_4-2/plot_4-2-1/h1hand-bookshelf_hard-v0/h1hand-bookshelf_hard-v0.pdf",
]

labels = ["(a) Cubes-v0", "(b) Powerlift-v0", "(c) Bookshelf_simple-v0", "(d) Bookshelf_hard-v0"]

output_path = "mujoco-benchmark/paintting/humanoidbench-paintting/plot_4-2/plot_4-2-1.pdf"
rows = 1  # 可以根据需要调整行数
cols = 4  # 可以根据需要调整列数

merge_pdfs_with_labels_dynamic(pdf_files, labels, output_path, rows, cols, dpi=300, label_font_size=18, spacing=30)
