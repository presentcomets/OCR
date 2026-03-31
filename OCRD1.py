import os
import io
import re
import pandas as pd
from google.cloud import vision
from thefuzz import process, fuzz
import pycountry
import argparse
import google.generativeai as genai
import json
from PIL import Image
import requests
from collections import OrderedDict
import warnings

from matplotlib import pyplot as plt
import cv2
import time
import math

warnings.filterwarnings("ignore")

#----------------------------------------------------------------------
os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS', 'key_1.json')
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "AIzaSyDKL3CLPoocySKYWYsRq18xyFOaUvQhW50"))
model = genai.GenerativeModel('gemini-2.5-flash')
#----------------------------------------------------------------------
#yolo_model = YOLO("best.pt")
#----------------------------------------------------------------------

# ---------------------------------------------------------
def get_platform_column(filename):
    fname = filename.lower()
    if 'watson' in fname: return 'Sales_Watson' 
    if 'shopee' in fname: return 'Sales_Shopee'
    if 'lazada' in fname: return 'Sales_Lazada' 
    if 'tiktok' in fname: return 'Sales_Tiktok'
    if 'konvy' in fname: return 'Sales_Konvy'
    return 'Sales_Other' 

def load_history_data(filename):
    existing_set =set()
    if os.path.exists(filename):
        try: 
            df = pd.read_excel(filename , dtype=str, engine='openpyxl')
            fda_col = next((c for c in df.columns if 'fda' in c.lower()), None)
            if fda_col:
                nums = df[fda_col].astype(str).str.replace(r'\D', '', regex=True).tolist()
                existing_set = set(nums)
        except: pass
    return existing_set

def update_sales_in_final_file(filename, fda_number, new_sales_val, target_col):
    if not os.path.exists(filename):
        return
    try:
        df_old = pd.read_excel(filename, dtype=str, engine='openpyxl')
        clean_target = re.sub(r'\D', '', str(fda_number))

        temp_col = 'Clean_Num_Temp'
        fda_col = next((c for c in df_old.columns if 'fda' in c.lower()), None)
        
        if fda_col:
            df_old[temp_col] = df_old[fda_col].astype(str).str.replace(r'\D', '', regex=True)
            mask = df_old[temp_col] == clean_target
            
            if mask.any():
                if target_col not in df_old.columns: 
                    df_old[target_col] = "-"
    
                if str(new_sales_val).strip() and str(new_sales_val).lower() not in ["nan", "", "-"]:
                    df_old.loc[mask, target_col] = str(new_sales_val)
                    print(f"   🔄 DUPLICATE: อัปเดต {target_col} เป็น {new_sales_val}")
                else:
                    print(f"   ℹ️ DUPLICATE: FDA {fda_number} ซ้ำแต่ไม่มียอดขาย")

            if temp_col in df_old.columns: del df_old[temp_col]
            
            df_old.to_excel(filename, index=False, engine='openpyxl')
            
    except Exception as e:
        print(f"   ❌ Update Error: {e}")


def analyze_color_and_mood(image_content):
    # ค่าเริ่มต้น (ถ้า AI พังจริงๆ ให้คืนค่านี้)
    default_data = {
        "Main_color": "-", "Hexcolor": "-", "Pictures": "-", 
        "Claims_Dct": "-", "Active_Claim": "-", "PerActiveClaim": "-", "Symbols": "-", "Claims": "-"
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            img = Image.open(io.BytesIO(image_content))
        
            prompt = """
            Analyze this product image. Return ONLY a raw JSON object.
            Extract:
            {
            "Main_color": "สีหลักและสีรองของผลิตภัณฑ์ 2 สี เช่น ขาว , ดำ" ,
            "Hexcolor": "ค่า Hex ของสีหลัก สีรอง (ถ้ามี) บนบรรจุภัณฑ์ 2 สี เช่น #FFFFFF, #000000",
            "Pictures": "รูปภาพที่ปรากฏบนบรรจุภัณฑ์ (เช่น ดอกไม้ ผลไม้ สัตว์) อธิบายสั้นๆ ง่ายๆ",
            "Claims" : "บอกว่าบนฉลาดมีการใช้คำเคลมว่าอะไร เช่น ลดสิว ผิวกระจ่างใส ลดริ้วรอย กันแดด เพิ่มความชุ่มชื้น ฯลฯ (บอกเป็นรายการคอมม่าเซปาเรท) ",
            "Claims_Dct" : "วิเคราะห์คำเคลม (Claims)" และจัดกลุ่มเป็น Categories หลัก คือ 
                        (
                            'Refining' :การปรับผิวให้เรียบเนียน กระชับรูขุมขน และฟื้นฟูสภาพผิวให้ดูสม่ำเสมอและสุขภาพดี  ,
                            'Acne treatment' :เน้นการ รักษาสิว ที่มีอยู่ในปัจจุบัน (Active Acne)  ,
                            'Anti-acne' : เน้นการ ป้องกันการเกิดสิว (Prevent Acne)  ,
                            'Anti-aging' : ลดเลือนริ้วรอย ชะลอความเสื่อมของผิว และฟื้นฟูผิวให้ดูอ่อนเยาว์ , 
                            'Anti-bacterial' : ฆ่าเชื้อแบคทีเรียหรือยับยั้งการเจริญเติบโตของแบคทีเรีย ช่วยป้องกันการเกิดสิว , 
                            'Anti-inflammatory' : ลดการอักเสบ บวมแดง และระคายเคืองบนผิว , 
                            'Antioxidant' : ป้องกันผิวจากอนุมูลอิสระ ลดความเสียหายจากมลภาวะและแสงแดด ช่วยให้ผิวแข็งแรง , 
                            'Brightening' : เน้นการปรับสีผิวให้ ดูสว่าง กระจ่างใส โดยการลดเลือนความหมองคล้ำและฟื้นฟูความสดใสของผิว , 
                            'Anti-dark spot' : ลดเลือนจุดด่างดำบนผิวที่เกิดจากการสะสมของเม็ดสีเมลานิน เช่น รอยสิว ฝ้า กระ , 
                            'Exfoliation' : การผลัดเซลล์ผิวเก่าออก ช่วยให้ผิวเรียบเนียน สดใส และกระจ่างใสขึ้น , 
                            'Firming' : การกระชับผิว ทำให้ผิวดูตึงและยกกระชับมากขึ้น ช่วยลดการหย่อนคล้อย , 
                            'Hydrating' : ผลิตภัณฑ์ที่ช่วย เพิ่มน้ำเข้าสู่ผิว เพื่อให้ผิวดูอิ่มน้ำและสดใส , 
                            'Moisturizing' : ผลิตภัณฑ์ที่ช่วย ป้องกันการสูญเสียน้ำในผิว โดยการสร้างชั้นเคลือบบนผิว เพื่อเก็บกักความชุ่มชื้นที่มีอยู่ในผิว , 
                            'Oil control' : ควบคุมการผลิตน้ำมันส่วนเกินบนผิว ช่วยป้องกันความมันส่วนเกินที่อาจทำให้เกิดสิวและความหมองคล้ำ , 
                            'Pore minimizing' : ลดขนาดรูขุมขนที่มองเห็นได้ ช่วยให้ผิวดูเนียนเรียบและกระชับขึ้น , 
                            'Skin repair' : ฟื้นฟูและซ่อมแซมผิวที่เสียหาย เช่น รอยแผลเป็นจากสิวหรือแสงแดด , 
                            'Skin rejuvenation' :  ฟื้นฟูผิวให้ดูอ่อนเยาว์ เพิ่มความกระจ่างใสและลดริ้วรอย , 
                            'UV Protection' : ปกป้องผิวจากรังสีอัลตราไวโอเลต (UV) ที่อาจทำให้เกิดความเสียหาย เช่น ริ้วรอย ฝ้า กระ หรือมะเร็งผิวหนัง  ,
                            'Anti-wrinkle' : ลดเลือนริ้วรอย , 
                            'Whitening' : เน้นให้ผิว ขาวขึ้น ด้วยการลดการผลิตเม็ดสีเมลานินในผิว , 
                            'Soothing' : ลอบประโลมและลดการระคายเคืองของผิว ช่วยทำให้ผิวสงบลงจากการอักเสบ แพ้ หรือระคายเคือง , 
                            'Anti-Dandruff' : ขจัดรังแค ป้องกันรังแค Even skin tone ช่วยให้สีผิวสม่ำเสมอ ,  Anti-malodor ลดกลิ่นไม่พึงประสงค์ , 
                            'Anti-Melasma' : ลดริ้วรอย และจุดด่างดำ จากฝ้า   Blemish ลดรอยด่างดำ ตำหนิต่างๆ เช่น สิว จุดด่างดำ รอยดำ/รอยแดง ฝ้า/กระ รูขุมขน , 
                            'Anti-Dandruff' : ลดรังแค , 
                            'Anti-Pollution' : ปกป้องผิวจากฝุ่นละอองและมลภาวะ เช่น PM 2.5  ,
                            'Blue Light Protection (HEV)' : ปกป้องผิวจากแสงสีฟ้าจากหน้าจอคอมพิวเตอร์และมือถือ ,
                            'Deep Hydration'/ '72h Moisture' : เติมความชุ่มชื้นล้ำลึก หรือกักเก็บความชุ่มชื้นยาวนาน (ระบุเวลา) ) 
                        โดย Category แค่ในกลุ่มนี้เท่านั้น
                        และระบุคำศัพท์ที่พบบนฉลากใส่ในวงเล็บ เช่น 
                            - Brightening (เช่น กระจ่างใส, ลดจุดด่างดำ, ออร่า, glow)
                            - Anti-Aging (เช่น ลดริ้วรอย, ยกกระชับ, หน้าเด็ก, ตีนกา)
                            - Moisturizing (เช่น ชุ่มชื้น, อิ่มน้ำ, ผิวแห้ง, เติมน้ำ)
                        แล้วคำนวณสัดส่วน (Calculation): นับจำนวนคำเคลมทั้งหมดที่เจอ (Total Claims) - คำนวณ % ของแต่ละกลุ่ม: (จำนวนคำในกลุ่ม / จำนวนทั้งหมด) * 100 - ปัดเศษทศนิยมให้เป็นจำนวนเต็ม
                        \nตอบในรูปแบบ: 'Category [คำที่พบ 1, คำที่พบ 2] ; XX%' คั่นแต่ละกลุ่มด้วยคอมม่า และเรียงจาก % มากไปน้อย 
                        เช่น Anti-Acne[ลดสิว,สิวผด]:35% ; Brightening[กระจ่างใส,ลดจุดด่างดำ]:10% ; Moisturizing[ชุ่มชื้น,อิ่มน้ำ]:8%" , 
            PerClaims_Dct" : "สัดส่วน % ของแต่ละกลุ่ม Claims_Dct เช่น Anti-Acne:35% , Brightening:10% , Moisturizing:8%"
            "Active_Claim" : "สารที่มีการบอกบนฉลาก เช่น Niacinamide , Hyaluronic Acid, Vit C ชื่อสารภาษาอังกฤษเท่านั้น (ถ้ามีหลายตัวให้คอมม่าเซปาเรท)"
            "PerActiveClaim" : "ปริมาณความเข้มข้นหรือระดับความแรงของสารที่ระบุบนฉลาก (เก็บหมดทั้งหน่วย % และ จำนวนเท่า) เช่น 'Niacinamide 5%', 'Vit C 50X', 'Hyaluron 30%'"
            "Symbols": "ระบุชื่อสัญลักษณ์มาตรฐานบนบรรจุภัณฑ์ (Packaging Symbols) เช่น Recycle, Cruelty Free, Halal, Paraben Free, Vegan, Green Dot (ระบุเฉพาะชื่อสัญลักษณ์)
                        ถ้าไม่เจอสัญลักษณ์ใดๆ ตอบ '-'
                        \n**สำคัญ:** สำหรับสัญลักษณ์อายุการใช้งาน (รูปกระปุกเปิดฝา) ให้ตอบในรูปแบบ 'PAO(ตัวเลขM)' เช่น PAO(12M), PAO(6M), PAO(24M)" 
            }

            Strictly JSON only. No markdown.
            """
        
            print("   🤖 AI: กำลังเจาะจงข้อมูลจากรูปภาพ...")
            response = model.generate_content(
                [prompt, img],
                generation_config={"response_mime_type": "application/json"}
            )
        
            text_response = response.text
            match = re.search(r'\{.*\}', text_response, re.DOTALL)
        
            if match:
                clean_json = match.group(0) # เอาเฉพาะส่วนที่เป็น JSON จริงๆ
                result = json.loads(clean_json)
                print("   ✅ AI: แกะข้อมูลสำเร็จ!")   
                print(f"      -> สี: {result.get('Main_color')}")
                print(f"      -> สัญลักษณ์: {result.get('Symbols')}")
                return {**default_data, **result}
            return default_data
            
        except Exception as e:
            print(f"   ⚠️ AI Error (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:  # ครั้งสุดท้าย
                print(f"   ❌ AI FAILED after {max_retries} attempts")
                return default_data
            time.sleep(2)  # รอก่อน retry
    
    return default_data

def process_single_product(links, existing_fdas, output_filename ,excel_brand="-", excel_product="-", raw_contents=None):
    full_text_accumulator = []
    first_image_content = None

    # รวม image bytes จาก raw_contents (ถ่ายรูป/อัปโหลดรูป) กับ links (URL) เข้าด้วยกัน
    all_contents = []
    if raw_contents:
        all_contents.extend(raw_contents)
    for link in links:
        content = download_image(link)
        if content:
            all_contents.append(content)

    for i, content in enumerate(all_contents):
        if i == 0: first_image_content = content

        try:
            client = vision.ImageAnnotatorClient()
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            if response.text_annotations:
                full_text_accumulator.append(response.text_annotations[0].description)
            if response.error.message:
                print(f"   ⚠️ API OCR Error: {response.error.message}")
        except Exception as e:
            print(f"   ❌ OCR Exception: {e}")
        

    combined_full_text = "\n\n--- E ---\n\n".join(full_text_accumulator)
    fda_regex = r'\b(?:\d{1,2}\s?-\s?\d\s?-\s?\d{7,10}|\d{2}\s?-\s?\d\s?-\s?\d\s?-\s?\d\s?-\s?\d{7})\b'
    if not re.search(fda_regex, combined_full_text):
        print("   ⚠️ SKIP: ไม่พบเลข FDA ในข้อความ OCR → ข้ามสินค้านี้")
        return None
 
    extracted_data = extract_cosmetic_info(combined_full_text)
    if extracted_data.get("FDA_Status") == "❌ ไม่เจอในฐานข้อมูล":
        if excel_brand and excel_brand not in ["-", "nan", ""]:
            if extracted_data.get("Brand", "-") in ["-", "nan", ""]:
                extracted_data["Brand"] = excel_brand
                print(f"   📋 Fallback Brand: ใช้จาก Excel → {excel_brand}")

        if excel_product and excel_product not in ["-", "nan", ""]:
            if extracted_data.get("Product_Name", "-") in ["-", "nan", ""]:
                extracted_data["Product_Name"] = excel_product
                print(f"   📋 Fallback Product: ใช้จาก Excel → {excel_product}")

    current_fda = re.sub(r'\D', '', str(extracted_data.get("FDA_Number", "")))
    if len(current_fda) >= 10 and current_fda in existing_fdas:
        return {"is_duplicate": True, "FDA_Number": extracted_data["FDA_Number"]}
    if os.path.exists(output_filename) and len(current_fda) >= 10:
        try:
            df_check = pd.read_excel(output_filename, dtype=str, engine='openpyxl')
            fda_col = next((c for c in df_check.columns if 'fda' in c.lower()), None)
            
            if fda_col:
                existing_nums = df_check[fda_col].astype(str).str.replace(r'\D', '', regex=True).tolist()
                if current_fda in existing_nums:
                    print(f"   ⏭️ SKIP: FDA {extracted_data['FDA_Number']} ซ้ำ (File)")
                    return {"is_duplicate": True, "FDA_Number": extracted_data["FDA_Number"]}
        except:
            pass    
    if first_image_content:

        print("   🤖 New Item: เริ่มวิเคราะห์ AI...")
        ai_data = analyze_color_and_mood(first_image_content)
        
        keys_to_merge = ["Main_color", "Hexcolor", "Pictures", "Claims_Dct", "Active_Claim", "PerActiveClaim", "Claims", "Symbols"]
        
        for key in keys_to_merge:
            ai_value = str(ai_data.get(key, "-")).strip()
            current_value = extracted_data.get(key, "-")
            if ai_value and ai_value != "-":
                if current_value and current_value != "-":
                    extracted_data[key] = merge_data(str(current_value), str(ai_value))
                else:
                    extracted_data[key] = ai_value
    
    extracted_data["is_duplicate"] = False
    return extracted_data
##------------------------------------------------------
COUNTRY_LIST = []
for c in pycountry.countries:
    COUNTRY_LIST.append(c.name)
    if hasattr(c, 'common_name'):
        COUNTRY_LIST.append(c.common_name)
    if hasattr(c, 'official_name'):
        COUNTRY_LIST.append(c.official_name)
COUNTRY_LIST = sorted(list(set(COUNTRY_LIST)), key=len, reverse=True)

Thai_maping_country = {
    "ไทย": "Thailand", "อังกฤษ": "United Kingdom", "สหรัฐอเมริกา": "United States",
    "ญี่ปุ่น": "Japan", "จีน": "China", "เกาหลีใต้": {"South Korea", "Republic of Korea","Korea"},
    "ฝรั่งเศส": "France", "เยอรมนี": "Germany", "อิตาลี": "Italy",
    "สเปน": "Spain", "รัสเซีย": "Russia", "แคนาดา": "Canada",
    "ออสเตรเลีย": "Australia", "อินเดีย": "India", "เวียดนาม": "Vietnam",
    "มาเลเซีย": "Malaysia", "สิงคโปร์": "Singapore", "อินโดนีเซีย": "Indonesia",
}
#-----------------------------------------------------------
def load_csv_robust(filename):
    paths = [filename, os.path.join('Data', 'dict', filename)]
    for p in paths:
        if os.path.exists(p):
            try: return pd.read_csv(p, dtype=str, encoding='utf-8-sig')
            except: 
                try: return pd.read_csv(p, dtype=str, encoding='cp874')
                except: continue
    return pd.DataFrame()
       #------------------------------------- โหลด inci
inci_db = load_csv_robust('Ingredients.csv')
if not inci_db.empty:
    col = 'Ingredient_Name' if 'Ingredient_Name' in inci_db.columns else inci_db.columns[1]
    inci_list = inci_db[col].dropna().tolist()
else:
    inci_list = []
print(f"โหลด Ingredients: {len(inci_list)} รายการ")
       #------------------------------------- โหลด inci
fda_db = load_csv_robust('FDA_ALL.csv')
if not fda_db.empty:
    fda_db = fda_db.rename(columns={'Product_EN': 'Product_Name', 'Brand_EN': 'Brand_EN', 'Man_': 'Man' , 'Import_': 'Imp'})
    fda_db['Clean_Num'] = fda_db['Number'].astype(str).str.replace(r'\D', '', regex=True)
print(f"โหลด FDA Database: {len(fda_db)} รายการ")
       #------------------------------------- โหลด inci
seven_db = load_csv_robust('711.csv')
claim_list = []
active_claim_list_711 = []

if not seven_db.empty:
    col_claim = 'Claim' if 'Claim' in seven_db.columns else seven_db.columns[1]
    col_active = next((c for c in seven_db.columns if 'active' in str(c).lower() and 'claim' in str(c).lower()), None)
    if not col_active and len(seven_db.columns) > 2: col_active = seven_db.columns[2]
    col_fda = next((c for c in seven_db.columns if 'fda' in str(c).lower() or 'เลข' in str(c)), None)

    if col_fda:
        seven_db['Clean_Num'] = seven_db[col_fda].astype(str).str.replace(r'\D', '', regex=True)
    
    claim_list = seven_db[col_claim].dropna().unique().tolist()
    if col_active:
        active_claim_list_711 = seven_db[col_active].dropna().unique().tolist()

print(f"โหลด 7-11: {len(seven_db)} รายการ")

#-----------------------------------------------------------
def match_fda(fda_number):
    clean_number = re.sub(r'\D', '', str(fda_number))
    
    if len(clean_number) < 10 or len(clean_number) > 13 or fda_db.empty: 
        return None
    
    match_row = fda_db[fda_db['Clean_Num'] == clean_number]
    
    if not match_row.empty:
        data = match_row.iloc[0]
        
        origin = "-"
        if len(clean_number) == 12:
            origin = "-" 
        elif len(clean_number) >= 3:
            origin_map = {'1': "ผลิต", '2': "นำเข้า", '3': "ส่งออก"}
            origin = origin_map.get(clean_number[2], "ไม่ทราบ")
        
        return {
            "FDANumber": data.get('Number', '-'),
            "Brand": str(data.get('Brand_EN', '-')),      
            "Product_Name": str(data.get('Product_Name', '-')),   
            "Manufacturer": str(data.get('Man', '-')),
            "Importer": str(data.get('Imp', '-')),
            "Origin_Type": origin
        }
    
    return None

#-----------------------------------------------------------

def find_brand_smart(full_text):
    if fda_db.empty or not full_text: 
        return None
    all_brands = fda_db['Brand_EN'].dropna().unique().tolist()
    all_brands = [b for b in all_brands if len(str(b).strip()) > 2]
    text_upper = full_text.upper()
    found_candidates = []
    for brand in all_brands:
        brand_upper = str(brand).upper()
        idx = text_upper.find(brand_upper)
        if idx != -1:
            found_candidates.append((idx, len(brand), brand))
    if found_candidates:
        found_candidates.sort(key=lambda x: (x[0], -x[1]))
        return {"Brand": found_candidates[0][2], "Score": 100}
    return None

#-----------------------------------------------------------
def extract_and_verify_ingredients(full_text):
    if not full_text or not inci_list: 
        return "-", "-", "-"

    pattern = re.compile(r'(ingredients|ส่วนประกอบ|ส่วนผสม| ingredient :)', re.IGNORECASE)
    match = pattern.search(full_text)
    if not match: return "-", "-", "-"
    
    start_idx = match.end()
    remaining_text = full_text[start_idx:]
    
    stop_keywords = ["วิธีใช้", "usage", "direction", "to use", "ผลิตโดย", "manufactured", "manufacturer", "จัดจำหน่าย", "distributor", "distributed", "นำเข้า", "imported", "importer", "คำเตือน", "Disclaimer",
                      "warning", "caution", "made in", "ผลิตใน", "ปริมาณสุทธิ", "net weight",]
    stop_pattern = re.compile(r'(' + '|'.join(stop_keywords) + r')', re.IGNORECASE)

    stop_match = stop_pattern.search(remaining_text)
    ingredient_text = remaining_text[:stop_match.start()] if stop_match else remaining_text
    
    raw_ingredient_text = ingredient_text.strip()
    
    cleaned = re.sub(r'[\r\n]+', ',', ingredient_text)
    cleaned = re.sub(r'[\/\|]', ',', cleaned)
    cleaned = re.sub(r'\d+\.', '', cleaned) # ลบเลขลำดับ 1. 2.
    
    raw_items = [x.strip() for x in cleaned.split(',') if len(x.strip()) > 2]

    verified_ingredients = []
    rejected_ingredients = []
    for item in raw_items:
        # ลบตัวเลขหรือจุดข้างหน้า (bullet points)
        clean_item = re.sub(r'^[\d\-\.\•\●\*\+\s]+', '', item).strip()
        
        if not clean_item: continue

        # 🔥 Fuzzy Match กับฐานข้อมูล
        best_match = process.extractOne(clean_item, inci_list, scorer=fuzz.token_sort_ratio)

        if best_match and best_match[1] >= 75 :
            verified_ingredients.append(best_match[0])
            
        else: 
            # 🔥 เกณฑ์ที่ 2 (ของใหม่): กรองก่อนทิ้งลง Rejected 🔥
            
            # 2.1 ต้องไม่มีภาษาไทย (ก-๙)
            has_thai = re.search(r'[\u0E00-\u0E7F]', clean_item)
            
            # 2.2 ต้องมีความยาวพอสมควร (> 3 ตัวอักษร)
            is_long_enough = len(clean_item) > 3
            
            # 2.3 ต้องมีตัวอักษรภาษาอังกฤษ (ไม่ใช่เลขล้วน หรือสัญลักษณ์ล้วน)
            has_english_char = re.search(r'[a-zA-Z]', clean_item)
            
            # 2.4 ต้องไม่ใช่คำทั่วไปที่ไม่ใช่สาร (Stop words แบบเข้มข้น)
            bad_words = ["water", "aqua", "parfum", "fragrance"] # ยกเว้นไว้ถ้าอยากเก็บ แต่ปกติพวกนี้จะ match เจออยู่แล้ว
            is_bad_word = clean_item.lower() in ["no", "free", "net", "vol", "g", "ml", "non", "and"]

            if not has_thai and is_long_enough and has_english_char and not is_bad_word:
                # ถ้าผ่านด่านทั้งหมด ค่อยเก็บไว้ดูต่างหน้า
                rejected_ingredients.append(clean_item)
            
            # ถ้าไม่ผ่าน (เช่น เป็นภาษาไทย, เป็นเลข, สั้นเกิน) -> ปล่อยทิ้งไปเลย (Drop)

    verified_text = ", ".join(verified_ingredients) if verified_ingredients else "-"
    rejected_text = ", ".join(rejected_ingredients) if rejected_ingredients else "-" 
    
    return raw_ingredient_text, verified_text, rejected_text
#-----------------------------------------------------------
def extract_claims_from_711(full_text):
    if not claim_list or not full_text: return "-"
    claims_output = []
    full_text_lower = full_text.lower()
    
    for claim in claim_list:
        claim_clean = claim.lower()
        if claim_clean in full_text_lower:
            claims_output.append(claim)
            continue
        ratio = fuzz.partial_ratio(claim_clean, full_text_lower)
        if ratio >= 80:
            claims_output.append(claim)
    seen = set()
    unique_claims = []
    for item in claims_output:
        
        if item.lower() not in seen:
            seen.add(item.lower())
      
            unique_claims.append(item.title()) 
            
    return ", ".join(unique_claims) if unique_claims else "-"
    
#------------------------------------------------------------------------------
#==============================================================================
# --- ฟังก์ชันที่ 1: Active Claim แบบ 7-11 (Human Curated) ---
#==============================================================================
def extract_active_claims_711(full_text):
    if not active_claim_list_711 or not full_text: return "-"
    
    full_text_lower = full_text.lower()
    found_actives = []
    
    for active_claim in active_claim_list_711:
        clean_active = str(active_claim).lower()
        if clean_active in full_text_lower:
            found_actives.append(active_claim)
            continue
        ratio = fuzz.partial_ratio(clean_active, full_text_lower)
        if ratio >= 85: 
            found_actives.append(active_claim)

    seen = set()
    unique_actives = []
    for item in found_actives:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique_actives.append(item.title())
            
    return ", ".join(unique_actives) if unique_actives else "-"

#==============================================================================
# --- ฟังก์ชันที่ 2: Active Claim แบบ Logic (Algorithm/Scientific) ---
#==============================================================================
def extract_active_claims_algorithm(full_text, raw_ingredient_text):
    if not inci_list or not full_text: return "-"
    
    marketing_text = full_text
    # ตัดส่วนผสมดิบออก เพื่อหาแค่ในคำโฆษณา
    if raw_ingredient_text != "-":
        marketing_text = full_text.replace(raw_ingredient_text, "")
    
    marketing_text_lower = marketing_text.lower()
    found_actives = []
    # กรองเอาเฉพาะคำที่ยาวเกิน 3 ตัวอักษร เพื่อลด Noise
    target_incis = [i for i in inci_list if len(str(i)) > 3]
    
    for inci in target_incis:
        if str(inci).lower() in marketing_text_lower:
            found_actives.append(inci)
            
    seen = set()
    unique_actives = []
    for item in found_actives:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique_actives.append(item.title())
            
    return ", ".join(unique_actives) if unique_actives else "-"
#------------------------------------------------------------------------------
def extract_field_until_next_keyword(full_text, field_name):
    field_keywords = {
        "Usage": ["วิธีใช้" , "วัธีใช้", "usage", "direction", "to use", "how to use"],
        "Disclaimer": ["คำเตือน", "ค่าเตือน", "คำแนะนำ", "ข้อควรระวัง", "warning", "caution", "precaution"],
        "Manufacturer": ["ผลิตโดย", "ชื่อผู้ผลิต", "manufactured by", "manufacturer", "made by"],
        "Distributor": ["จัดจำหน่ายโดย", "distributor", "distributed by" , "ผู้จัดจำหน่าย"],
        "Importer": ["นำเข้าโดย", "imported by", "importer", "ผู้นำเข้า"],
        "Made_in": ["made in", "ผลิตใน", "country of origin"],
        "Net_Weight": ["ปริมาณสุทธิ", "net weight", "net wt", "net vol", "net content"],
        "Product_Type": ["ประเภทผลิตภัณฑ์", "product type", "category"]
    }
    keywords = field_keywords.get(field_name, [])
    if not keywords: return None
    all_other_keywords = []
    for fname, kws in field_keywords.items():
        if fname != field_name: all_other_keywords.extend(kws)
    all_other_keywords.extend(["ingredients", "ส่วนประกอบ", "ส่วนผสม"])
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        match = pattern.search(full_text)
        if match:
            start_idx = match.end()
            remaining = full_text[start_idx:]
            min_stop_idx = len(remaining)
            for stop_kw in all_other_keywords:
                stop_pattern = re.compile(re.escape(stop_kw), re.IGNORECASE)
                stop_match = stop_pattern.search(remaining)
                if stop_match and stop_match.start() < min_stop_idx:
                    min_stop_idx = stop_match.start()
            extracted = remaining[:min_stop_idx].strip()
            extracted = re.sub(r'^[\s:\-—]+', '', extracted)
            
            # 🔥 สำหรับ Net_Weight และ Made_in ตัดให้สั้นลง
            if field_name in ["Net_Weight", "Made_in"]:
                # ตัดที่ขึ้นบรรทัดใหม่ หรือ --- E --- หรือเกิน 100 ตัวอักษร
                lines = extracted.split('\n')
                extracted = lines[0].strip()  # เอาบรรทัดแรกเท่านั้น
                
                # ตัดที่เจอ --- E ---
                if '--- E ---' in extracted:
                    extracted = extracted.split('--- E ---')[0].strip()
                
                # สำหรับ Net_Weight: ตัดให้เหลือแค่ 30 ตัวอักษรแรก
                if field_name == "Net_Weight":
                    extracted = extracted[:30].strip()
                
                # สำหรับ Made_in: ตัดให้เหลือแค่ 50 ตัวอักษรแรก
                elif field_name == "Made_in":
                    extracted = extracted[:50].strip()
            
            if extracted:
                return extracted
    return None
#------------------------------------------------------------------------------
def clean_country(text):
    """
    แปลงเป็นชื่อประเทศ โดยใช้เฉพาะ Thai_maping_country
    ตัดข้อความยาวๆ ออก เอาเฉพาะชื่อประเทศ
    """
    if not text or text == "-": 
        return "-"
    
    # ตัดเอาเฉพาะ 50 ตัวอักษรแรก เพื่อไม่ให้ข้อความยาวเกิน
    text = str(text)[:50].strip()
    
    # ลบอักขระพิเศษ
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    
    # เช็คกับ mapping ที่กำหนดไว้
    for th_name, en_name in Thai_maping_country.items():
        if th_name in clean_text:
            # กรณีที่เป็น set (เกาหลีใต้)
            if isinstance(en_name, set):
                return list(en_name)[0]
            return en_name
    
    # เช็คภาษาอังกฤษ (case-insensitive)
    clean_lower = clean_text.lower()
    for th_name, en_name in Thai_maping_country.items():
        en_check = en_name if not isinstance(en_name, set) else list(en_name)[0]
        if en_check.lower() in clean_lower:
            return en_check
    
    # ถ้าไม่เจอใน mapping ให้คืน "-"
    return "-"
#------------------------------------------------------------------------------
def PAO_symbol(text):
    if not text: return "-"
    patterns = [
        (r'(?<!\d)\b(\d{1,2})\s*[Mm]\b', 1) 
    ]
    for p, group_idx in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            # ดึงตัวเลขตาม Group Index ที่ระบุไว้คู่กัน
            number = m.group(group_idx)
            return f"{number}M" # คืนค่าเป็นฟอร์แมตมาตรฐาน เช่น 12M
            
    return None
#------------------------------------------------------------------------------
def Sun_Protection(text):
    if not text: return "-"
    
    spf_match = re.search(r'\bSPF\s*[:\-]?\s*(\d{1,3}(?:\+)?)(?:\s|$|[^a-zA-Z0-9])', text, re.IGNORECASE)
    spf_text = f"SPF {spf_match.group(1)}" if spf_match else ""

    pa_match = re.search(r'\b(PA\s*\+{1,4})', text, re.IGNORECASE)
    
    pa_text = ""
    if pa_match:
        pa_text = pa_match.group(1).replace(" ", "").upper()

    results = []
    if spf_text: results.append(spf_text)
    if pa_text: results.append(pa_text)
    
    return " ".join(results) if results else "-"
#------------------------------------------------------------------------------
def validate_brand_with_db(extracted_brand):
    if not extracted_brand or str(extracted_brand).strip() == "" or str(extracted_brand).strip() == "-":
        return extracted_brand
    if fda_db.empty:
        return extracted_brand
    unique_brands = fda_db['Brand_EN'].dropna().unique().tolist()
    try:
        best_match = process.extractOne(str(extracted_brand), unique_brands, scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] >= 80: return best_match[0]
    except: pass
    return extracted_brand
#------------------------------------------------------------------------------
def parse_quantity(text):
    """
    แปลงน้ำหนักเป็นหน่วยมาตรฐาน ml หรือ g เท่านั้น
    ดึงเฉพาะตัวเลข+หน่วยที่ตรงกัน ไม่เอาข้อความที่เหลือ
    """
    if not text:
        return None
    
    # เอาแค่ 100 ตัวอักษรแรกเพื่อหาน้ำหนัก (ป้องกันข้อความยาวเกิน)
    text = str(text)[:100]
    
    # Pattern สำหรับ ml (มล, ML, mL ทั้งหมดแปลงเป็น ml)
    ml_pattern = r'(\d{1,4}(?:[.,]\d+)?)\s*(ml|mL|ML|มล\.?|มิลลิลิตร)\b'
    ml_match = re.search(ml_pattern, text, flags=re.IGNORECASE)
    if ml_match:
        number = ml_match.group(1).replace(',', '.')
        return f"{number} ml"
    
    # Pattern สำหรับ g (กรัม, gram, Gram ทั้งหมดแปลงเป็น g)
    g_pattern = r'(\d{1,4}(?:[.,]\d+)?)\s*(g\b|G\b|gm\b|g\.|กรัม|gram\b|Gram\b)(?!\s*k)'
    g_match = re.search(g_pattern, text, flags=re.IGNORECASE)
    if g_match:
        number = g_match.group(1).replace(',', '.')

        return f"{number} g"
    
    # Pattern สำหรับ kg
    kg_pattern = r'(\d{1,4}(?:[.,]\d+)?)\s*(kg|kilo|kilogram|กิโลกรัม)\b'
    kg_match = re.search(kg_pattern, text, flags=re.IGNORECASE)
    if kg_match:
        number = kg_match.group(1).replace(',', '.')
        return f"{number} kg"
    
    # Pattern สำหรับ sheets/pcs
    sheets_pattern = r'(\d{1,4}(?:[.,]\d+)?)\s*(sheets|sheet|pcs|แผ่น|ชิ้น)\b'
    sheets_match = re.search(sheets_pattern, text, flags=re.IGNORECASE)
    if sheets_match:
        number = sheets_match.group(1).replace(',', '.')
        return f"{number} sheets"
    
    return None
#------------------------------------------------------------------------------
def predict_side(full_text, filename=""):
    filename_lower = filename.lower()
    #if "front" in filename_lower: return "Front"
    #if "back" in filename_lower: return "Back"
    #if "side" in filename_lower: return "Side"
    #if not full_text: return "Front"

    full_text_lower = full_text.lower()
    score = 0

    weak_keywords = ["วิธีใช้", "usage", "direction", "to use", "ปริมาณสุทธิ", "net weight", "net vol", "คำเตือน", "warning"]
    strong_keywords = ["ingredients", "ส่วนประกอบ", "ส่วนผสม", "ผลิตโดย", "manufactured", "จัดจำหน่าย", "distributor", "นำเข้าโดย", "imported", "เลขที่ใบรับแจ้ง", "เลขที่ใบรับจดแจ้ง", "mfg", "exp"]
    
    for kw in weak_keywords:
        if kw in full_text_lower: score += 1

    for kw in strong_keywords:
        if kw in full_text_lower: score += 3

    if len(full_text) > 300: score += 2
    return "Back" if score >= 4 else "Front"
#------------------------------------------------------------------------------
def merge_data(db_text, ai_text):
    if not db_text or db_text == "-": return ai_text
    if not ai_text or ai_text == "-": return db_text
    # ถ้าเหมือนกันเป๊ะ ให้ใช้อันเดียว
    if db_text.strip().lower() == ai_text.strip().lower(): return db_text
    # ถ้ารวมแล้วยาวไป เช็คซ้ำนิดนึง
    if ai_text.lower() in db_text.lower(): return db_text
    return f"{db_text}, {ai_text}"
#------------------------------------------------------------------------------
def extract_cosmetic_info(full_text, filename=""):
    full_text = full_text or ""
    data = {
        "FDA_Number": "-", "FDA_Status": "-", "TypeFDA": "-",
        "Brand": "-", "Product_Name": "-",
        "Claims": "-", "Claims_Dct": "-", "Active_Claim": "-", "PerActiveClaim": "-",
        "Ingredients_Raw": "-", "Ingredients_Verified": "-", "Ingredients_Rejected": "-",
        "Main_color": "-", "Hexcolor": "-", "Pictures": "-", "Key_Ingredient": "-", 
        "Net_Weight": "-", "Sun_Protection": "-", "PAO_Symbol": "-",
        "Product_Type": "-", "Usage": "-", "Disclaimer": "-", "Manufacturer": "-",
        "Made_in": "-", "Distributor": "-", "Importer": "-" , "Symbols" : "-", "FDA_Status711": "-", "Full_Text": full_text, 
    }
    
    #data["Side"] = predict_side(full_text, filename)
    
    fda_regex = r'\b(?:\d{1,2}\s?-\s?\d\s?-\s?\d{7,10}|\d{2}\s?-\s?\d\s?-\s?\d\s?-\s?\d\s?-\s?\d{7})\b'
    fda_match = re.search(fda_regex, full_text)
    if fda_match:
        data["FDA_Number"] = re.sub(r'\s+', '', fda_match.group(0))
        clean_num = re.sub(r'\D', '', data["FDA_Number"])
        if len(clean_num) >= 3:
             origin_map = {'1': "ผลิต", '2': "นำเข้า", '3': "ส่งออก"}
             data["TypeFDA"] = origin_map.get(clean_num[2], "-")
        fda_info = match_fda(data["FDA_Number"])
        if fda_info:
            data["FDA_Status"] = "✅ เจอในฐานข้อมูล"
            data.update({
                "Brand": fda_info["Brand"],
                "Product_Name": fda_info["Product_Name"],
                "Manufacturer": fda_info["Manufacturer"],
                "TypeFDA": fda_info["Origin_Type"]
            })
        else:
            data["FDA_Status"] = "❌ ไม่เจอในฐานข้อมูล"

    if data["Brand"] == "-":
        brand_info = find_brand_smart(full_text)
        if brand_info: data["Brand"] = brand_info["Brand"]
    
    if data["Brand"] == "-":
        lines = [l.strip() for l in full_text.splitlines() if l.strip()]
        for ln in lines[:3]:
            if len(ln) > 2 and not any(k in ln.lower() for k in ["วิธีใช้", "ingredients", "ส่วนประกอบ", "ผลิตโดย"]):
                data["Brand"] = ln
                break

    fields = ["Usage", "Disclaimer", "Manufacturer", "Distributor", "Importer", "Made_in", "Net_Weight", "Product_Type"]
    for field in fields:
        val = extract_field_until_next_keyword(full_text, field)
        
        if val: 
            if field == "Net_Weight":
                parsed = parse_quantity(val)
                if parsed: 
                    data["Net_Weight"] = parsed
            
            elif field == "Made_in":
                country = clean_country(val)
                if country != "-":
                    data["Made_in"] = country
            
            else:
                data[field] = val
    ing_raw, ing_ver, ing_rejected = extract_and_verify_ingredients(full_text)
    data["Ingredients_Raw"] = ing_raw 
    data["Ingredients_Verified"] = ing_ver
    data["Ingredients_Rejected"] = ing_rejected 

    ai_claim = extract_claims_from_711(full_text)
 
    data["Claims"] = ai_claim

    current_fda = re.sub(r'\D', '', data.get("FDA_Number", ""))
    
    # เช็คว่า seven_db มีข้อมูลและมีคอลัมน์ Clean_Num
    if not seven_db.empty and 'Clean_Num' in seven_db.columns:
        match_row = seven_db[seven_db['Clean_Num'] == current_fda]
        
        if not match_row.empty:
            row = match_row.iloc[0]
            
            # ดึงข้อมูลจาก DB
            col_claim = 'Claim' if 'Claim' in seven_db.columns else seven_db.columns[1]
            db_claim = str(row[col_claim]).strip()
            
            if db_claim.lower() != 'nan':
                # ✅ เจอ: เอามาใส่
                data["Claims"] = db_claim
                data["FDA_Status711"] = "✅ เจอในฐานข้อมูล 7-11"
            
            # ถ้าไม่เจอ (else): ปล่อย data["Claim"] เป็น "-" ตามค่าเริ่มต้น

    # Other detections
    data["Sun_Protection"] = Sun_Protection(full_text)
    data["PAO_Symbol"] = PAO_symbol(full_text)
    
    # ❌ ลบบรรทัด data["Active_Claim_711"] = ... ออกด้วย
    
    return data
#------------------------------------------------------------------------------
def download_image(url):
    try:
        url = url.strip()
        if not url: return None
        if url.startswith(('http://', 'https://')):
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                return response.content
        elif os.path.exists(url):
            with open(url, 'rb') as f:
                return f.read()
    except: pass
    return None
#------------------------------------------------------------------------------
def merge_image_page(image_contents, max_width_per_img=800):
    if not image_contents or len(image_contents) < 1:
        return None

    images = []
    for content in image_contents:
        try:
            img = Image.open(io.BytesIO(content)).convert('RGB')
            w_percent = max_width_per_img / float(img.size[0])
            h_size = int(img.size[1] * w_percent)
            img = img.resize((max_width_per_img, h_size), Image.Resampling.LANCZOS)
            images.append(img)
        except Exception as e:
            print(f"   ⚠️ ข้ามรูปภาพที่อ่านไม่ได้: {e}")

    if not images:
        return None

    cols    = 2 if len(images) > 1 else 1
    rows    = math.ceil(len(images) / cols)
    grid_w  = cols * max_width_per_img
    grid_h  = sum(
        max(images[r * cols + c].size[1]
            for c in range(cols) if r * cols + c < len(images))
        for r in range(rows)
    )

    combined_image = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    y_offset = 0                         

    for r in range(rows):
        row_max_h = 0
        for c in range(cols):
            idx = r * cols + c
            if idx < len(images):
                img      = images[idx]
                x_offset = c * max_width_per_img
                combined_image.paste(img, (x_offset, y_offset))
                row_max_h = max(row_max_h, img.size[1])
        y_offset += row_max_h          

    # ── แปลงเป็น bytes ────────────────────────────────────────
    output_buf = io.BytesIO()
    combined_image.save(output_buf, format='JPEG', quality=85)
    print()
    return output_buf.getvalue()  
#------------------------------------------------------------------------------
#def detect_symbols_with_yolo(image_content):
    if yolo_model is None: return []
    try:
        img = Image.open(io.BytesIO(image_content))
        results = yolo_model(img, verbose=False)
        detected = []
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) > 0.7: 
                    cls_id = int(box.cls[0])
                    detected.append(yolo_model.names[cls_id])
        return list(set(detected))
    except:
        return []
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def save_checkpoint(new_data, filename="OCScos.xlsx"):
    if not new_data: return
    
    df_new = pd.DataFrame(new_data)
    
    # จัดเรียงคอลัมน์ให้สวยงาม
    cols = ["Source_Links","FDA_Number", "FDA_Status" , "TypeFDA", "Brand", "Product_Name", 
            "Claims","Claims_Dct","Active_Claim", "PerActiveClaim",
            "Ingredients_Raw","Ingredients_Verified" , "Ingredients_Rejected", "Key_Ingredient","Sun_Protection",
            "Net_Weight","PAO_Symbol", "Symbols" ,"Main_color", "Hexcolor", "Pictures", "Product_Type",
            "Usage", "Disclaimer","Made_in", "Manufacturer", "Distributor", "Importer" , "FDA_Status711" , "Sales_Watson", 
            "Sales_Shopee", "Sales_Lazada", "Sales_Tiktok", "Sales_Konvy", "Sales_Eveandboy", "Sales_Other"]
    final_cols = [c for c in cols if c in df_new.columns] + [c for c in df_new.columns if c not in cols]
    df_new = df_new[final_cols]

    if os.path.exists(filename):
        try:
            
            df_old = pd.read_excel(filename, dtype=str, engine='openpyxl')
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.to_excel(filename, index=False, engine='openpyxl')
        except:
            
            df_new.to_excel(filename, index=False, engine='openpyxl')
    else:
        
        df_new.to_excel(filename, index=False, engine='openpyxl')
        
    print(f"   💾 Auto-Saved: บันทึกข้อมูลเพิ่ม {len(new_data)} รายการ ลงไฟล์สำเร็จ")

#------------------------------------------------------------------------------
def process_excel_file(file_path):
    if os.path.basename(file_path).startswith("~$"): return []
    output_filename = "OCRcos1.xlsx"
    
    current_platform_col = get_platform_column(os.path.basename(file_path))
    print(f"   🎯 แพลตฟอร์ม: {current_platform_col}")

    existing_fdas = load_history_data(output_filename)
    print(f"   📚 ประวัติเดิม: {len(existing_fdas)} รายการ")
    
    print(f"   📄 อ่านไฟล์: {os.path.basename(file_path)}")
    try:
        df = pd.read_excel(file_path, dtype=str, engine='openpyxl')
        df = df.fillna("")
        df.columns = [str(c).strip() for c in df.columns]
        
        img_col = None
        sales_col = None
        brand_col = None    
        product_col = None 
        for col in df.columns:
            clean_col_name = str(col).strip().lower() 
            
            if 'url รูปเพิ่มเติม' in clean_col_name or 'image' in clean_col_name or 'picture' in clean_col_name: 
                img_col = col

            if any(k in clean_col_name for k in ['sold', 'sale', 'ขาย', 'ยอดขาย']): 
                sales_col = col
            # เช็คคอลัมน์แบรนด์
            if 'brand' in clean_col_name: 
                brand_col = col
            # เช็คคอลัมน์ชื่อสินค้า
            if any(k in clean_col_name for k in ['product', 'สินค้า', 'ชื่อสินค้า']): 
                product_col = col
        if img_col:
            for idx, row in df.iterrows():
                raw_val = str(row[img_col]).strip()
                if not raw_val or raw_val.lower() == 'nan': continue
                
                current_sales = str(row[sales_col]).strip() if sales_col else "-"
                
                clean_val = re.sub(r',\s*(http)', r' \1', raw_val.replace('|', ' ').replace('\n', ' '))
                all_links = [l.strip() for l in clean_val.split() if l.strip().startswith('http')]
                unique_links = list(OrderedDict.fromkeys(all_links))
                
                if not unique_links: continue

                print(f"\n📦 สินค้าลำดับที่ {idx+1}: มี {len(unique_links)} รูป")
                # ใหม่
                # ... (code เก่า: print สินค้าลำดับที่...) ...

                # 🔥🔥🔥 [แทรกใหม่ตรงนี้] ดึงค่าจาก Excel เตรียมไว้ 🔥🔥🔥
                excel_brand   = str(row[brand_col]).strip()   if brand_col   else "-"
                excel_product = str(row[product_col]).strip() if product_col else "-"
                result = process_single_product(unique_links, existing_fdas, output_filename, excel_brand, excel_product)
                if result is None:
                    print("   🗑️ ไม่มี FDA → ข้ามไปสินค้าถัดไป")
                    continue

                if result.get("is_duplicate"):
                    update_sales_in_final_file(output_filename, result["FDA_Number"], current_sales, current_platform_col)
                    time.sleep(0.5) 
   
                elif result:
                    result["Source_Links"] = raw_val
                    result[current_platform_col] = current_sales 
                    
                    save_checkpoint([result], output_filename)
                    
                    new_fda = re.sub(r'\D', '', str(result.get("FDA_Number", "")))
                    if new_fda: existing_fdas.add(new_fda)
                    
                    print("   ⏳ รอ 30 วินาที...")
                    time.sleep(5)
        else:
            print("      ⚠️ ไม่พบคอลัมน์รูปภาพ")
            
    except Exception as e:
        if "KeyboardInterrupt" not in str(e): print(f"      ❌ Error: {e}")
#------------------------------------------------------------------------------
def process_image_folder(image_paths):
    if not image_paths: return
    output_filename = "OCRcos1.xlsx"  
    existing_fdas = load_history_data(output_filename)
    valid_images = ('.jpg', '.jpeg', '.png', '.webp')
    subfolders = [f for f in os.listdir(target_images) if os.path.isdir(os.path.join(target_images, f))]
    if sub_folders:
        # ✅ Mode ใหม่: แต่ละโฟลเดอร์ย่อย = 1 สินค้า
        sub_folders_data = []
        for folder_name in sub_folders:
            folder_path = os.path.join(base_folder, folder_name)
            img_paths = sorted([
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(valid_images)
            ])
            if img_paths:
                sub_folders_data.append((folder_name, img_paths))

    print(f"\n📸 พบไฟล์รูปภาพทั้งหมด {len(image_paths)} รูปในโฟลเดอร์")
    
    for idx, img_path in enumerate(image_paths):
        print(f"\n🖼️ ประมวลผลรูปที่ {idx+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        folder_name = os.path.basename(os.path.dirname(img_path))
        current_platform_col = get_platform_column(folder_name)
        
        result = process_single_product([img_path], existing_fdas, output_filename)
        
        if result is None:
            print("   🗑️ ไม่มี FDA → ข้ามไปรูปถัดไป")
            continue

        if result.get("is_duplicate"):
            print(f"   ⏭️ SKIP: FDA {result.get('FDA_Number')} ซ้ำในระบบแล้ว")
            continue
            
        elif result:
            result["Source_Links"] = img_path  
            result[current_platform_col] = "-" 
            
            save_checkpoint([result], output_filename)
            
            new_fda = re.sub(r'\D', '', str(result.get("FDA_Number", "")))
            if new_fda: existing_fdas.add(new_fda)
            
            print("   ⏳ รอ 5 วินาที...")
            time.sleep(5)
#------------------------------------------------------------------------------
def run_script(input_path):
    valid_excels = ('.xlsx', '.xls')
    valid_images = ('.jpg', '.jpeg', '.png', '.webp')
    target_excels = []
    target_images = []
    
    if os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if not file.startswith("~$"):
                    file_lower = file.lower()
                    if file_lower.endswith(valid_excels):
                        target_excels.append(os.path.join(root, file))
                    elif file_lower.endswith(valid_images):
                        target_images.append(os.path.join(root, file))
    else:
        file_lower = input_path.lower()
        if file_lower.endswith(valid_excels):
            target_excels.append(input_path)
        elif file_lower.endswith(valid_images):
            target_images.append(input_path)

    print("🚀 เริ่มต้นการทำงาน... (ระบบจะบันทึกอัตโนมัติทีละรายการ)")
    
    try:
        # ส่วนที่ 1: รันไฟล์ Excel (ถ้ามี)
        for file_path in target_excels:
            process_excel_file(file_path)
            
        # ส่วนที่ 2: รันไฟล์รูปภาพ (ถ้ามี)
        if target_images:
            process_image_folder(target_images)
    except KeyboardInterrupt:
        print("\n👋 จบการทำงาน (User Stopped)")
    
    print(f"\n✅ เสร็จสิ้น! ตรวจสอบไฟล์ final_results11.5.xlsx")
#------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="image")
    args = parser.parse_args()
    run_script(args.input)