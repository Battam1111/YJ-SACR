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

def merge_pdfs_with_labels_dynamic(pdf_files, labels, output_path, rows, cols, dpi=300):
    """将多张PDF图片合并为一张PDF，自动根据图片尺寸调整合并后的PDF页面大小，并在每张图片下方添加标签。"""
    
    # 获取所有PDF文件的尺寸（假设所有PDF文件的尺寸相同）
    pdf_width, pdf_height = get_pdf_page_size(pdf_files[0])
    
    # 计算合并后的PDF页面尺寸
    total_width = pdf_width * cols
    total_height = (pdf_height + 20) * rows  # 20单位的高度用于标签
    
    # 创建一个新的PDF文件，页面大小为动态计算的大小
    c = canvas.Canvas(output_path, pagesize=(total_width, total_height))
    
    # 计算每个子图的位置和大小
    num_pdfs = len(pdf_files)
    
    for i, (pdf_file, label) in enumerate(zip(pdf_files, labels)):
        # 计算该图在页面中的位置
        col = i % cols
        row = i // cols
        x_position = col * pdf_width
        y_position = total_height - (row + 1) * (pdf_height + 20)
        
        # 将PDF页面转换为图像，使用较高的DPI以提高分辨率
        images = convert_from_path(pdf_file, first_page=0, last_page=1, dpi=dpi)
        img = images[0]
        
        # 将图像保存为临时文件
        img_temp_path = f"temp_img_{i}.png"
        img.save(img_temp_path, 'PNG')
        
        # 在PDF页面上绘制图像
        c.drawImage(img_temp_path, x_position, y_position + 20, width=pdf_width, height=pdf_height)
        
        # 删除临时图像文件
        os.remove(img_temp_path)
        
        # 在图像下方添加标签，标签居中
        c.setFont("Helvetica", 18)
        label_width = c.stringWidth(label, "Helvetica", 18)  # 调整为和标签字体相同的大小
        label_x_position = x_position + (pdf_width - label_width) / 2  # 居中计算
        c.drawString(label_x_position, y_position + 5, label)
        
        # 如果已经填满一页，则新建一页
        if (i + 1) % (rows * cols) == 0:
            c.showPage()
    
    # 保存最终的合并PDF
    c.save()


# 使用示例
pdf_files = [ "mujoco-benchmark/PlotSet-FULL-NoRD-cvgspd/Humanoid/multiple/AF(AF0.2)/AF(AF0.2)_average_reward_smoothed.pdf",
              "mujoco-benchmark/PlotSet-FULL-NoRD-cvgspd/HumanoidStandup/multiple/AF(AF0.2)/AF(AF0.2)_average_reward_smoothed.pdf", 
              "mujoco-benchmark/PlotSet-FULL-NoRD-cvgspd/Hopper/multiple/AF(AF0.2)/AF(AF0.2)_average_reward_smoothed.pdf", 
              "mujoco-benchmark/PlotSet-FULL-NoRD-cvgspd/Walker2d/multiple/AF(AF0.2)/AF(AF0.2)_average_reward_smoothed.pdf", 
              "mujoco-benchmark/PlotSet-FULL-NoRD-cvgspd/HalfCheetah/multiple/AF(AF0.2)/AF(AF0.2)_average_reward_smoothed.pdf", 
              "mujoco-benchmark/PlotSet-FULL-NoRD-cvgspd/Reacher/multiple/AF(AF0.2)/AF(AF0.2)_average_reward_smoothed.pdf"]


# pdf_files = [ "mujoco-benchmark/PlotSet-FULL-NoRD-cvgspd/Humanoid/multiple/AF(AF0.2)/AF(AF0.2)_average_reward_convergence_speed_bar_chart.pdf",
#               "mujoco-benchmark/PlotSet-FULL-NoRD-cvgspd/HumanoidStandup/multiple/AF(AF0.2)/AF(AF0.2)_average_reward_convergence_speed_bar_chart.pdf", 
#               "mujoco-benchmark/PlotSet-FULL-NoRD-cvgspd/Hopper/multiple/AF(AF0.2)/AF(AF0.2)_average_reward_convergence_speed_bar_chart.pdf", 
#               "mujoco-benchmark/PlotSet-FULL-NoRD-cvgspd/Walker2d/multiple/AF(AF0.2)/AF(AF0.2)_average_reward_convergence_speed_bar_chart.pdf", 
#               "mujoco-benchmark/PlotSet-FULL-NoRD-cvgspd/HalfCheetah/multiple/AF(AF0.2)/AF(AF0.2)_average_reward_convergence_speed_bar_chart.pdf", 
#               "mujoco-benchmark/PlotSet-FULL-NoRD-cvgspd/Reacher/multiple/AF(AF0.2)/AF(AF0.2)_average_reward_convergence_speed_bar_chart.pdf"]



labels = ["(a) Humanoid-v4", "(b) HumanoidStandup-v4", "(c) Hopper-v4", "(d) Walker2d-v4", "(e) HalfCheetah-v4", "(f) Reacher-v4"]

output_path = "mujoco-benchmark/paintting/merged_AvgReward.pdf"
# output_path = "mujoco-benchmark/paintting/merged_convergeSpeed.pdf"

rows = 2
cols = 3

merge_pdfs_with_labels_dynamic(pdf_files, labels, output_path, rows, cols)