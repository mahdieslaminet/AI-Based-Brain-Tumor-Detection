# create_model.py
import tensorflow as tf
from tensorflow.keras import layers, models

def create_and_save_model():
    print("در حال دانلود وزن‌های اولیه MobileNetV2...")
    # استفاده از مدل از پیش آموزش دیده (Transfer Learning)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False # فریز کردن لایه‌های پایه

    # اضافه کردن لایه‌های طبقه بندی مخصوص پروژه ما
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='softmax') # 4 کلاس خروجی
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("مدل ساخته شد.")
    model.summary()
    
    # ذخیره مدل
    model.save('brain_tumor_model.h5')
    print("✅ فایل brain_tumor_model.h5 با موفقیت ذخیره شد.")
    print("حالا می‌توانید app.py را اجرا کنید.")

if __name__ == "__main__":
    create_and_save_model()
