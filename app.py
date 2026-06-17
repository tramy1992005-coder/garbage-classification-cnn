import io
import os
import base64
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from torchvision import transforms
from typing import List
import pdfkit

# --- 1. KIẾN TRÚC MẠNG RESNET CUSTOM (ĐỒNG BỘ 100% VỚI KAGGLE) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class AdvancedGarbageCNN(nn.Module):
    def __init__(self):
        super(AdvancedGarbageCNN, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# --- 2. KHỞI TẠO FASTAPI & CẤU HÌNH LIÊN KẾT CORS ---
app = FastAPI(title="Garbage Classification API - ResNet Custom")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. CẤU HÌNH SIÊU THAM SỐ & TẢI TRỌNG SỐ MÔ HÌNH ---
IMG_SIZE = 128
CLASSES = ['O', 'R']
GROUP_MAP = {'O': 'Organic (Rác hữu cơ)', 'R': 'Recyclable (Rác tái chế)'}

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdvancedGarbageCNN().to(device)

try:
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    print("-> Tải trọng số mô hình thành công!")
except Exception as e:
    print(f"-> LỖI KHỞI TẠO MÔ HÌNH: {e}")

# --- 4. PHỤC VỤ GIAO DIỆN WEB TRỰC TIẾP TỪ SERVER ---
@app.get("/")
async def read_index():
    return FileResponse("index.html")

# --- 5. API ENDPOINT 1: PHÂN TÍCH ẢNH ĐƠN LẺ ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        top_confidence, top_class_idx = torch.max(probabilities, 0)
        top_class = CLASSES[top_class_idx.item()]

        return {
            "top_class": top_class,
            "top_confidence": float(top_confidence.item()),
            "group": GROUP_MAP[top_class]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- 6. API ENDPOINT 2: PHÂN TÍCH NHANH HÀNG LOẠT TRÊN WEB ---
# --- Cập nhật lại API Endpoint Batch trong file app.py ---
@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                
            prob_o = float(probabilities[0])
            prob_r = float(probabilities[1])
            pred_class = CLASSES[0] if prob_o > prob_r else CLASSES[1]
            confidence = prob_o if pred_class == 'O' else prob_r
            
            results.append({
                "filename": file.filename,
                "prediction": pred_class,
                "confidence": confidence,
                "prob_o": prob_o,
                "prob_r": prob_r,
                "group": GROUP_MAP[pred_class]
            })
        except Exception:
            results.append({"filename": file.filename, "error": "Không thể xử lý ảnh"})
            
    return {"results": results}

# --- 7. API ENDPOINT 3: PHÂN TÍCH HÀNG LOẠT & IN BÁO CÁO PDF ---
@app.post("/predict-batch-pdf")
async def predict_batch_pdf(files: List[UploadFile] = File(...)):
    try:
        count_o, count_r = 0, 0
        items_html = ""
        
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            # Chuyển ảnh gốc của rác sang định dạng chuỗi Base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Chạy mô hình dự đoán phân phối xác suất
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            
            prob_o, prob_r = probabilities[0], probabilities[1]
            pred_class = CLASSES[0] if prob_o > prob_r else CLASSES[1]
            confidence = prob_o if pred_class == 'O' else prob_r
            
            if pred_class == 'O': count_o += 1
            else: count_r += 1
            
            # Tự động vẽ đồ thị thanh ngang phân phối xác suất lớp bằng Matplotlib
            fig, ax = plt.subplots(figsize=(3.8, 1.4))
            bars = ax.barh(['Organic (O)', 'Recyclable (R)'], [prob_o, prob_r], color=['#ea580c', '#2563eb'], height=0.5)
            ax.set_xlim(0, 1.0)
            ax.tick_params(axis='both', labelsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width*100:.1f}%', va='center', ha='left', fontsize=8, fontweight='bold')
            
            chart_buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(chart_buffer, format='png', dpi=180, bbox_inches='tight')
            plt.close()
            chart_base64 = base64.b64encode(chart_buffer.getvalue()).decode('utf-8')
            
            badge_class = "badge-o" if pred_class == 'O' else "badge-r"
            group_text = GROUP_MAP[pred_class]
            
            # Nạp dữ liệu cấu trúc HTML inline
            items_html += f"""
            <div class="result-item" style="display: block; width: 100%; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 15px; overflow: hidden; page-break-inside: avoid;">
                <div style="float: left; width: 25%; text-align: center; padding: 10px; background: #fafafa; border-right: 1px solid #e2e8f0;">
                    <img src="data:image/png;base64,{img_base64}" style="max-width: 110px; max-height: 110px; border-radius: 4px;">
                </div>
                <div style="float: left; width: 35%; padding: 10px;">
                    <div style="font-weight: bold; font-size: 11px; margin-bottom: 6px; color: #1e293b;">{file.filename}</div>
                    <span style="display: inline-block; padding: 2px 6px; font-size: 10px; font-weight: bold; border-radius: 4px; margin-bottom: 6px;" class="{badge_class}">{group_text}</span>
                    <div style="font-size: 10px; color: #64748b;">Độ tin cậy dự đoán:</div>
                    <div style="font-size: 15px; font-weight: bold;">{confidence*100:.1f}%</div>
                </div>
                <div style="float: right; width: 35%; text-align: center; padding: 10px;">
                    <img src="data:image/png;base64,{chart_base64}" style="max-width: 220px;">
                </div>
                <div style="clear: both;"></div>
            </div>
            """
        
        # Mẫu sườn cấu trúc PDF báo cáo tổng hợp
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #1e293b; line-height: 1.5; padding: 10px; }}
                .header {{ border-bottom: 2px solid #16a34a; padding-bottom: 10px; margin-bottom: 20px; }}
                .badge-o {{ background: #ffedd5; color: #c2410c; }}
                .badge-r {{ background: #dbeafe; color: #1d4ed8; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1 style="font-size: 22px; color: #16a34a; margin: 0; text-transform: uppercase;">Báo Cáo Phân Loại Rác Thải Hàng Loạt</h1>
                <p style="margin:4px 0 0 0; font-size:12px; color:#64748b;">Hệ thống nhận diện tự động - Mô hình ResNet Deep Learning nhị phân</p>
            </div>
            <div style="width: 100%; margin-bottom: 20px; background: #f8fafc; border: 1px solid #e2e8f0; font-size: 12px; padding: 10px; border-radius: 6px;">
                <b>Mô hình xử lý:</b> AdvancedGarbageCNN &nbsp;&nbsp;|&nbsp;&nbsp; <b>Tổng số lượng mẫu quét:</b> {len(files)} ảnh
            </div>
            <h3 style="font-size: 14px; border-left: 4px solid #16a34a; padding-left: 8px; margin-bottom: 15px;">Tổng quan phân bố số lượng</h3>
            <div style="width: 100%; margin-bottom: 25px; overflow: hidden;">
                <div style="float: left; width: 48%; background: #fff7ed; border: 1px solid #ffedd5; color: #c2410c; padding: 12px; text-align: center; border-radius: 6px;">
                    <div style="font-size:11px; font-weight:bold;">RÁC HỮU CƠ (ORGANIC)</div>
                    <div style="font-size:24px; font-weight:bold;">{count_o}</div>
                </div>
                <div style="float: right; width: 48%; background: #eff6ff; border: 1px solid #dbeafe; color: #1d4ed8; padding: 12px; text-align: center; border-radius: 6px;">
                    <div style="font-size:11px; font-weight:bold;">RÁC TÁI CHẾ (RECYCLABLE)</div>
                    <div style="font-size:24px; font-weight:bold;">{count_r}</div>
                </div>
            </div>
            <div style="clear: both; height: 10px;"></div>
            <h3 style="font-size: 14px; border-left: 4px solid #16a34a; padding-left: 8px; margin-bottom: 15px;">Kết quả phân tích chi tiết từng ảnh</h3>
            {items_html}
        </body>
        </html>
        """
        
        pdf_path = "output_batch_report.pdf"
        
        # Liên kết trực tiếp đến file thực thi wkhtmltopdf vừa setup trên Windows
        path_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
        config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
        
        # Xuất và biên dịch mã html sang tệp PDF vật lý
        pdfkit.from_string(html_template, pdf_path, configuration=config, options={"enable-local-file-access": ""})
        
        return FileResponse(pdf_path, filename="Bao_Cao_Phan_Loai_Rac_Hang_Loat.pdf", media_type="application/pdf")
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
# --- 8. KHỞI ĐỘNG SERVER UVICORN ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)