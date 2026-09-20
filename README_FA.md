# اجرای Benchmark طلای ۱۸ عیار با GitHub Actions

این بسته چهار مدل را **جداگانه** اجرا می‌کند و هیچ ensemble یا مدل اختصاصی نمی‌سازد:

1. Google TimesFM 3.0 — checkpoint رسمی `google/timesfm-3.0-pytorch`
2. Amazon Chronos-2 — checkpoint رسمی `amazon/chronos-2`
3. Salesforce Moirai 2.0 — checkpoint رسمی `Salesforce/moirai-2.0-R-small`
4. XGBoost Quantile Regression — benchmark/challenger

## نکته مهم
برای رایگان ماندن GitHub Actions، repository را **Public** بسازید. Runner استاندارد public رایگان است. مدل‌ها روی CPU اجرا می‌شوند، پس ممکن است چند ساعت طول بکشد. گوشی لازم نیست در تمام مدت روشن یا متصل بماند.

## نتیجه
در پایان workflow یک artifact به نام:

`gold18-frontier-final-results`

ساخته می‌شود که شامل `report.html`، leaderboardها، forecastهای فعلی، رکوردهای walk-forward و ZIP نهایی است.

## قانون انتخاب
برنده فقط مدلی است که هر سه افق 3M/6M/12M را کامل کند و کمترین میانگین `pinball loss` را داشته باشد. اگر یک مدل رسمی خطا بدهد، جای آن مدل دست‌ساز قرار داده نمی‌شود؛ خطا در گزارش ثبت می‌شود.

## داده
ابتدا تلاش می‌شود تاریخچه TGJU تازه شود. اگر دسترسی TGJU ممکن نباشد، فایل معتبر همراه بسته (`data/gold18_ohlc.csv`) استفاده می‌شود.

## مجوز TimesFM-3
وزن‌های pretrained نسخه 3 تحت مجوز non-commercial/non-production منتشر شده‌اند؛ این بسته برای benchmark پژوهشی طراحی شده است.
