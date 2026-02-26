from flask import Flask, render_template, request, jsonify
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# 모델 로드
MODEL_PATH = "elasticnet_bmd_pipeline.joblib"
META_PATH = "meta.json"

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully")
    else:
        model = None
        print(f"⚠️ Model file not found: {MODEL_PATH}")
    
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        print(f"📊 Meta loaded: {len(meta.get('num_cols', []))} numeric, {len(meta.get('cat_cols', []))} categorical")
    else:
        meta = None
        print(f"⚠️ Meta file not found: {META_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    meta = None

# =========================
# HTML 페이지 라우트
# =========================
@app.route('/')
def index():
    return render_template('habio-1-1-start.html')

@app.route('/survey')
def survey():
    return render_template('habio-1-2-survey.html')

@app.route('/signup')
def signup():
    return render_template('habio-1-3-signup.html')

@app.route('/home')
def home():
    return render_template('habio-2-1-home.html')

@app.route('/quests')
def quests():
    return render_template('habio-3-1-quests.html')

@app.route('/stack')
def stack():
    return render_template('habio-3-2-stack.html')

@app.route('/league')
def league():
    return render_template('habio-3-3-league.html')

@app.route('/report')
def report():
    return render_template('habio-3-4-report.html')

@app.route('/profile')
def profile():
    return render_template('habio-4-1-profile.html')

@app.route('/friends')
def friends():
    return render_template('habio-4-2-friends.html')

# =========================
# API: BDS 예측
# =========================
@app.route('/api/predict-bds', methods=['POST'])
def predict_bds():
    if model is None:
        return jsonify({'error': 'Model not loaded', 'bds': 72, 'category': 'Moderate'}), 200
    
    try:
        data = request.json
        
        # BMI 계산 (키와 몸무게로부터)
        height_m = float(data.get('height', 160)) / 100
        weight_kg = float(data.get('weight', 60))
        bmi = weight_kg / (height_m ** 2)
        
        # 설문 데이터를 CSV 컬럼명으로 매핑
        input_data = {
            # 기본 정보
            'age': int(data.get('age', 30)),
            'edu': 3.0,  # 기본값 (1~4)
            'HE_BMI': bmi,
            
            # 생활습관
            'BS1_1': 1 if data.get('smoking') == 'yes' else 0,  # 흡연
            'BD1_11': 1 if data.get('alcohol') == 'yes' else 0,  # 음주
            'E_Q_SUN': 2,  # 햇빛 노출 (기본값)
            
            # 질병력
            'DM2_lt': 0,  # 당뇨 (기본값)
            'DF2_lt': 1 if data.get('fracture') == 'yes' else 0,  # 골절
            'LQ4_00': 1 if data.get('rheumatoid') == 'yes' else 0,  # 류마티스
            'DX_Q_hsty': 1 if data.get('parent_osteo') == 'yes' else 0,  # 부모 골다공증
            
            # 호르몬 관련
            'BP5': 0,  # 기본값
            'BP8': 0,  # 기본값
            'BE3_11': 0,  # 기본값
            'BE5_1': 0,  # 기본값
            
            # 여성 특화
            'mens': 0 if data.get('menopause') == 'yes' else 1,  # 폐경이면 0, 아니면 1
            'estrog': 1 if data.get('estrogen') == 'yes' else 0,  # 에스트로겐
        }
        
        # DataFrame 생성
        df = pd.DataFrame([input_data])
        
        # 모델이 기대하는 컬럼 순서대로 정렬
        if meta and 'num_cols' in meta:
            expected_cols = meta['num_cols'] + meta.get('cat_cols', [])
            # 누락된 컬럼 추가
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = 0
            # 순서 맞추기
            df = df[[col for col in expected_cols if col in df.columns]]
        
        # 예측
        bmd_prediction = float(model.predict(df)[0])
        
        # T-score 계산: (BMD - 1.065) / 0.122
        t_score = (bmd_prediction - 1.065) / 0.122
        
        # BDS 점수로 변환 (T-score를 0-100 스케일로)
        # T-score 범위: 보통 -4 ~ +2 정도
        # -2.5 이하: 관리 필요 (0-40)
        # -2.5 ~ -1.0: 주의 (40-70)
        # -1.0 이상: 정상 (70-100)
        if t_score >= -1.0:
            # 정상: 70-100
            bds_score = min(100, int(70 + (t_score + 1.0) * 15))
            category = "Good"
        elif t_score >= -2.5:
            # 주의: 40-70
            bds_score = int(40 + (t_score + 2.5) * 20)
            category = "Moderate"
        else:
            # 관리 필요: 0-40
            bds_score = max(0, int(40 + (t_score + 2.5) * 16))
            category = "Caution"
        
        return jsonify({
            'success': True,
            'bds': bds_score,
            'category': category,
            'bmd_raw': round(bmd_prediction, 3),
            't_score': round(t_score, 2),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        # 에러 시 기본값 반환
        return jsonify({
            'success': False,
            'error': str(e),
            'bds': 72,
            'category': 'Moderate'
        }), 200

# =========================
# API: LS 점수 업데이트 (식단 제출)
# =========================
@app.route('/api/submit-meal', methods=['POST'])
def submit_meal():
    try:
        data = request.json
        
        # 식단 데이터에서 칼슘/비타민D 풍부 식품 체크
        meals = [
            data.get('breakfast', ''),
            data.get('snack_am', ''),
            data.get('lunch', ''),
            data.get('snack_pm', ''),
            data.get('dinner', ''),
            data.get('night', '')
        ]
        
        all_text = ' '.join(meals).lower()
        
        # 간단한 키워드 기반 점수 계산
        points = 0
        
        # 칼슘 풍부 식품
        calcium_foods = ['우유', '치즈', '요거트', '요구르트', '멸치', '두부', '브로콜리']
        for food in calcium_foods:
            if food in all_text:
                points += 5
                break
        
        # 비타민D 풍부 식품
        vitamin_d_foods = ['연어', '고등어', '계란', '달걀']
        for food in vitamin_d_foods:
            if food in all_text:
                points += 5
                break
        
        # 각 식사 기록마다 기본 점수
        filled_meals = sum(1 for m in meals if m.strip())
        points += filled_meals * 2
        
        return jsonify({
            'success': True,
            'points_earned': points,
            'message': f'+{points} LS 획득!'
        })
        
    except Exception as e:
        print(f"❌ Meal submission error: {e}")
        return jsonify({'error': str(e)}), 400

# =========================
# Health check
# =========================
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # 로컬 테스트용
    app.run(debug=True, host='0.0.0.0', port=5000)
