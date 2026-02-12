from django import forms


# ── 下拉選單選項（對齊訓練資料的實際值）──

EDUCATION_CHOICES = [
    ("", "-- 請選擇 --"),
    ("研究所以上", "研究所以上"),
    ("碩士", "碩士"),
    ("專科/大學", "專科/大學"),
    ("大學", "大學"),
    ("專科", "專科"),
    ("高中/職", "高中/職"),
    ("高中", "高中"),
    ("小學", "小學"),
    ("其他", "其他"),
]

RESIDENCE_STATUS_CHOICES = [
    ("", "-- 請選擇 --"),
    ("自有", "自有"),
    ("租屋", "租屋"),
    ("配偶", "配偶"),
    ("親屬", "親屬"),
    ("宿舍", "宿舍"),
]

MAIN_BUSINESS_CHOICES = [
    ("", "-- 請選擇 --"),
    ("製造業", "製造業"),
    ("服務業", "服務業"),
    ("商業", "商業"),
    ("科技業", "科技業"),
    ("金融業", "金融業"),
    ("保險業", "保險業"),
    ("證券及期貨業", "證券及期貨業"),
    ("公教人員", "公教人員"),
    ("軍人", "軍人"),
    ("餐飲業", "餐飲業"),
    ("運輸業", "運輸業"),
    ("營造業", "營造業"),
    ("不動產業", "不動產業"),
    ("倉儲業", "倉儲業"),
    ("通信業", "通信業"),
    ("水電燃氣業", "水電燃氣業"),
    ("農牧林漁", "農牧林漁"),
    ("漁業", "漁業"),
    ("礦業及土石採取業", "礦業及土石採取業"),
    ("專業人士", "專業人士"),
    ("自由業", "自由業"),
    ("網拍業", "網拍業"),
    ("學生", "學生"),
    ("家管", "家管"),
    ("社會團體即個人服務", "社會團體即個人服務"),
    ("其他", "其他"),
]

PRODUCT_CHOICES = [
    ("", "-- 請選擇 --"),
    # ("商品融資", "商品融資"),
    ("瘦身美容", "瘦身美容"),
    ("3C家電", "3C家電"),
    ("個人用品", "個人用品"),
    # ("婚友聯誼", "婚友聯誼"),
    ("其他", "其他"),
]


class PredictionForm(forms.Form):
    """違約預測輸入表單（欄位對齊 DPMPredictor 所需輸入）"""

    # ── 基本資料 ──
    education = forms.ChoiceField(
        label="教育程度",
        choices=EDUCATION_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    month_salary = forms.FloatField(
        label="月薪（元）",
        min_value=0,
        widget=forms.NumberInput(attrs={"class": "form-input", "placeholder": "例：45000"}),
    )
    job_tenure = forms.FloatField(
        label="工作年資（年）",
        min_value=0,
        widget=forms.NumberInput(attrs={"class": "form-input", "placeholder": "例：3.5", "step": "0.1"}),
    )

    # ── 居住與職業 ──
    residence_status = forms.ChoiceField(
        label="居住狀態",
        choices=RESIDENCE_STATUS_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    main_business = forms.ChoiceField(
        label="主要行業別",
        choices=MAIN_BUSINESS_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    product = forms.ChoiceField(
        label="借款目的",
        choices=PRODUCT_CHOICES,
        widget=forms.Select(attrs={"class": "form-select"}),
    )

    # ── 貸款資訊 ──
    loan_term = forms.IntegerField(
        label="貸款期數（月）",
        min_value=1,
        widget=forms.NumberInput(attrs={"class": "form-input", "placeholder": "例：36"}),
    )
    paid_installments = forms.IntegerField(
        label="已繳期數",
        min_value=0,
        widget=forms.NumberInput(attrs={"class": "form-input", "placeholder": "例：12"}),
    )
    debt_to_income_ratio = forms.FloatField(
        label="負債收入比",
        min_value=0,
        widget=forms.NumberInput(attrs={"class": "form-input", "placeholder": "例：0.35（35%）", "step": "0.01"}),
        help_text="0.35 代表 35%",
    )
    payment_to_income_ratio = forms.FloatField(
        label="還款收入比",
        min_value=0,
        widget=forms.NumberInput(attrs={"class": "form-input", "placeholder": "例：0.15（15%）", "step": "0.01"}),
        help_text="0.15 代表 15%",
    )

    # ── 郵遞區號 ──
    post_code_permanent = forms.IntegerField(
        label="戶籍地郵遞區號",
        widget=forms.NumberInput(attrs={"class": "form-input", "placeholder": "例：100"}),
    )
    post_code_residential = forms.IntegerField(
        label="居住地郵遞區號",
        widget=forms.NumberInput(attrs={"class": "form-input", "placeholder": "例：106"}),
    )

    # ── 逾期紀錄 ──
    overdue_before_first = forms.IntegerField(
        label="第一個月前逾期次數",
        min_value=0,
        initial=0,
        widget=forms.NumberInput(attrs={"class": "form-input"}),
    )
    overdue_first_half = forms.IntegerField(
        label="第一個月上半月逾期次數",
        min_value=0,
        initial=0,
        widget=forms.NumberInput(attrs={"class": "form-input"}),
    )
    overdue_first_second_half = forms.IntegerField(
        label="第一個月下半月逾期次數",
        min_value=0,
        initial=0,
        widget=forms.NumberInput(attrs={"class": "form-input"}),
    )
    overdue_month_2 = forms.IntegerField(
        label="第二個月逾期次數",
        min_value=0,
        initial=0,
        widget=forms.NumberInput(attrs={"class": "form-input"}),
    )
    overdue_month_3 = forms.IntegerField(
        label="第三個月逾期次數",
        min_value=0,
        initial=0,
        widget=forms.NumberInput(attrs={"class": "form-input"}),
    )
    overdue_month_4 = forms.IntegerField(
        label="第四個月逾期次數",
        min_value=0,
        initial=0,
        widget=forms.NumberInput(attrs={"class": "form-input"}),
    )
    overdue_month_5 = forms.IntegerField(
        label="第五個月逾期次數",
        min_value=0,
        initial=0,
        widget=forms.NumberInput(attrs={"class": "form-input"}),
    )
    overdue_month_6 = forms.IntegerField(
        label="第六個月逾期次數",
        min_value=0,
        initial=0,
        widget=forms.NumberInput(attrs={"class": "form-input"}),
    )

    def clean_paid_installments(self):
        paid = self.cleaned_data.get("paid_installments")
        loan_term = self.cleaned_data.get("loan_term")
        if paid is not None and loan_term is not None and paid > loan_term:
            raise forms.ValidationError("已繳期數不能超過貸款期數。")
        return paid
