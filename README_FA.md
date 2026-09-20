# Benchmark چندمتغیره طلای ۱۸ عیار — v2

این نسخه همان چهار مدل منتشرشده را **جداگانه** اجرا می‌کند و هیچ ensemble یا مدل اختصاصی نمی‌سازد:

1. Google TimesFM 3.0 — `google/timesfm-3.0-pytorch`
2. Amazon Chronos-2 — `amazon/chronos-2`
3. Salesforce Moirai 2.0 — `Salesforce/moirai-2.0-R-small`
4. XGBoost Quantile Regression — challenger

## ورودی‌ها
- Target: log price طلای ۱۸ عیار
- Past-only covariate 1: log USD/IRR (دلار آزاد TGJU، `price_dollar_rl`)
- Past-only covariate 2: log XAU/USD (اونس جهانی TGJU، `ons`)

هم‌ترازی فقط با backward as-of روی تاریخ طلای ۱۸ عیار انجام می‌شود؛ بنابراین برای هر تاریخ فقط آخرین دلار/اونس موجود در همان تاریخ یا قبل از آن استفاده می‌شود و هیچ future covariate به مدل داده نمی‌شود. اگر دلار یا اونس TGJU قابل دریافت نباشد، benchmark چندمتغیره عمداً FAIL می‌شود و به Gold-only برنمی‌گردد.

## بک‌تست
پنجره‌های ۱۲ماهه غیرهمپوشان، افق‌های 3M/6M/12M، معیار اصلی mean pinball loss برای Q10/Q50/Q90. برنده فقط مدلی است که هر سه افق را کامل کند.

## XGBoost
همان objective رسمی `reg:quantileerror` استفاده می‌شود. ورودی فقط 63 lag خام بازده برای Gold18/USD/XAU است؛ هیچ اندیکاتور تکنیکال یا feature اختصاصی اضافه نشده است.

## مجوز
وزن‌های TimesFM-3 برای non-commercial/non-production منتشر شده‌اند؛ این benchmark پژوهشی است.
