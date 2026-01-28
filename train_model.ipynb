import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# --- تنظیمات ---
DATASET_PATH = 'dataset'  # نام پوشه‌ای که عکس‌ها داخلش هستند
IMG_SIZE = (224, 224)     # سایز استاندارد ورودی MobileNet
BATCH_SIZE = 32           # تعداد عکس‌هایی که در هر دور پردازش می‌شوند
EPOCHS = 10               # تعداد دفعاتی که کل دیتاست مرور می‌شود (برای تست ۱۰ کافیست)

def train_brain_tumor_model():
    print("--- شروع فرآیند آماده‌سازی داده‌ها ---")
    
    # بررسی وجود پوشه دیتاست
    if not os.path.exists(DATASET_PATH):
        print(f"❌ خطا: پوشه '{DATASET_PATH}' پیدا نشد!")
        print("لطفاً پوشه Training دیتاست را در کنار این فایل قرار دهید و نام آن را به dataset تغییر دهید.")
        return

    # 1. ساخت Data Generator (پیش‌پردازش و افزایش داده)
    # ما ۲۰٪ داده‌ها را برای اعتبارسنجی (Validation) جدا می‌کنیم
    datagen = ImageDataGenerator(
        rescale=1./255,          # نرمال‌سازی پیکسل‌ها بین 0 و 1
        validation_split=0.2,    # ۲۰ درصد برای تست حین آموزش
        rotation_range=20,       # کمی چرخش برای یادگیری بهتر (Data Augmentation)
        horizontal_flip=True     # برعکس کردن افقی
    )

    # لود کردن داده‌های آموزشی
    print("در حال بارگذاری داده‌های آموزشی (Train)...")
    train_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # لود کردن داده‌های اعتبارسنجی
    print("در حال بارگذاری داده‌های اعتبارسنجی (Validation)...")
    val_generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # چاپ نام کلاس‌ها برای اطمینان
    print(f"کلاس‌های شناسایی شده: {train_generator.class_indices}")

    # 2. ساخت مدل (Transfer Learning با MobileNetV2)
    print("--- در حال ساخت مدل ---")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # فریز کردن لایه‌های پایه برای سرعت بیشتر

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='softmax')  # 4 کلاس خروجی
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 3. شروع آموزش (Training)
    print(f"--- شروع آموزش مدل برای {EPOCHS} دور (Epoch) ---")
    print("این مرحله بسته به قدرت سیستم شما ممکن است چند دقیقه تا چند ساعت طول بکشد...")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator
    )

    # 4. ذخیره مدل نهایی
    print("--- آموزش تمام شد. در حال ذخیره مدل ---")
    model.save('brain_tumor_model.h5')
    print("✅ فایل brain_tumor_model.h5 با موفقیت آپدیت شد.")
    print("حالا می‌توانید app.py را اجرا کنید و از هوش مصنوعی واقعی لذت ببرید!")

if __name__ == "__main__":
    train_brain_tumor_model()



