# 4.3 Hasil Pre-processing Data

Pada penelitian ini, tahap pre-processing dilakukan langsung di dalam pipeline pelatihan model multimodal. Data teks dibaca dari kolom `Teks Terlihat`, sedangkan data gambar diambil dari folder `dataset/fix-review-manual-label`. Implementasi menunjukkan bahwa pre-processing manual yang dilakukan peneliti relatif minimal, karena sebagian proses telah ditangani oleh komponen model, yaitu tokenizer ELECTRA untuk modalitas teks dan `CLIPImageProcessor` untuk modalitas gambar. Dengan pendekatan ini, data dapat langsung disiapkan dalam format yang sesuai untuk proses pelatihan tanpa memerlukan tahapan pre-processing manual yang terlalu kompleks.

## 4.3.1 Pre-processing Modalitas Gambar

Pada modalitas gambar, setiap file citra terlebih dahulu dibuka dari direktori dataset lalu dikonversi ke format RGB. Setelah itu, pada data latih dilakukan augmentasi untuk meningkatkan variasi data dan membantu model menjadi lebih robust terhadap perubahan visual yang wajar. Augmentasi hanya diterapkan pada data training (`is_train=True`), sedangkan data validasi tidak diberi augmentasi agar evaluasi model tetap merepresentasikan data asli.

Augmentasi yang digunakan dalam implementasi ini terdiri atas tiga transformasi. Pertama, `RandomHorizontalFlip()` untuk membalik gambar secara horizontal secara acak. Kedua, `RandomRotation(degrees=15)` untuk melakukan rotasi acak hingga 15 derajat. Ketiga, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` untuk memberikan variasi pada kecerahan, kontras, saturasi, dan hue. Kombinasi augmentasi ini bertujuan memperkaya distribusi visual data tanpa mengubah makna utama gambar secara ekstrem.

Setelah augmentasi dilakukan, gambar diproses menggunakan `CLIPImageProcessor` dari model `openai/clip-vit-base-patch32`. Pada tahap ini, gambar diubah menjadi `pixel_values` yang siap masuk ke encoder visual CLIP. Sesuai konsep pada Subbab 3.2.7, normalisasi nilai piksel dan penyesuaian format input tidak dilakukan secara manual, melainkan otomatis oleh CLIP processor sebagai bagian dari pipeline model.

Potongan kode yang digunakan adalah sebagai berikut:

```python
if self.is_train:
    self.augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
    ])
else:
    self.augmentation = None

image = Image.open(img_path).convert('RGB')

image_for_clip = image
if self.is_train and self.augmentation:
    image_for_clip = self.augmentation(image)

clip_inputs = self.clip_processor(images=image_for_clip, return_tensors='pt')
pixel_values = clip_inputs['pixel_values'].squeeze(0)
```

Berdasarkan implementasi tersebut, dapat disimpulkan bahwa pre-processing gambar pada penelitian ini dilakukan melalui dua tahap utama, yaitu augmentasi data pada data latih dan normalisasi otomatis oleh CLIP processor. Dengan demikian, data visual yang masuk ke model telah berada dalam representasi yang sesuai dengan kebutuhan encoder CLIP.

## 4.3.2 Pre-processing Modalitas Teks

Pada modalitas teks, data diambil dari kolom `Teks Terlihat` pada file CSV. Implementasi tidak menunjukkan adanya tahap pembersihan teks secara manual seperti case folding eksplisit, stopword removal, stemming, atau lemmatization. Sebaliknya, pendekatan yang digunakan adalah menyerahkan proses pre-processing langsung kepada tokenizer bawaan model ELECTRA yang dipanggil melalui `AutoTokenizer`.

Tokenizer yang digunakan dimuat dari model `sentinet/suicidality`, yang pada pipeline ini berperan sebagai tokenizer untuk encoder teks ELECTRA. Teks kemudian diproses dengan beberapa parameter penting, yaitu `truncation=True` untuk memotong teks yang terlalu panjang, `padding='max_length'` untuk menyamakan panjang seluruh input, `max_length=128` untuk membatasi panjang token maksimum, serta `return_tensors='pt'` agar hasil tokenisasi langsung berbentuk tensor PyTorch.

Dari proses ini dihasilkan dua komponen utama, yaitu `input_ids` dan `attention_mask`. `input_ids` merupakan representasi numerik dari token hasil tokenisasi, sedangkan `attention_mask` menandai token yang valid dan token hasil padding. Dengan demikian, sesuai Subbab 3.2.7, proses tokenisasi, penyesuaian format input, dan pemetaan teks ke representasi numerik dilakukan secara otomatis oleh tokenizer ELECTRA sebagai bagian dari pipeline model.

Potongan kode yang digunakan adalah sebagai berikut:

```python
text = str(row['Teks Terlihat']) if 'Teks Terlihat' in row else ''

enc = self.electra_tokenizer(
    text,
    truncation=True,
    padding='max_length',
    max_length=self.max_len,
    return_tensors='pt'
)

input_ids = enc['input_ids'].squeeze(0)
attention_mask = enc['attention_mask'].squeeze(0)
```

Selain itu, inisialisasi tokenizer pada program utama dilakukan dengan kode berikut:

```python
electra_tokenizer = AutoTokenizer.from_pretrained(config.ELECTRA_MODEL_NAME)
```

Berdasarkan implementasi tersebut, pre-processing teks pada penelitian ini lebih banyak ditangani oleh tokenizer model daripada dilakukan secara manual. Pendekatan ini membuat proses pemrosesan teks menjadi lebih efisien dan konsisten dengan kebutuhan model transformer yang digunakan.

## 4.3.3 Pembagian Dataset

Pembagian dataset dilakukan pada file `src/main.py`. Pertama, data dibaca dari file `fix-review-manual-label.csv`, kemudian hanya label yang termasuk dalam dua kelas utama, yaitu `NON-SELF-HARM` dan `SELF-HARM`, yang dipertahankan. Setelah itu, label kategorikal diubah menjadi label numerik, dengan `NON-SELF-HARM = 0` dan `SELF-HARM = 1`.

Selanjutnya, dataset dibagi menggunakan fungsi `train_test_split()` dari `scikit-learn` dengan rasio 80:20. Parameter `random_state=42` digunakan agar pembagian data bersifat konsisten pada setiap eksperimen, sedangkan `stratify=labels_df['Label']` digunakan untuk menjaga proporsi kelas pada data latih dan data validasi tetap serupa dengan distribusi data awal. Berdasarkan implementasi yang ada, pembagian data dilakukan menjadi dua subset, yaitu data training dan data validation. Dengan kata lain, pada pipeline saat ini belum terdapat pemisahan test set terpisah, sehingga evaluasi akhir dilakukan menggunakan data validasi.

Hasil pembagian dataset pada implementasi ini adalah sebagai berikut:

| Subset | Total Data | NON-SELF-HARM | SELF-HARM |
|---|---:|---:|---:|
| Keseluruhan dataset | 2178 | 1525 (70,02%) | 653 (29,98%) |
| Training | 1742 | 1220 (70,03%) | 522 (29,97%) |
| Validation | 436 | 305 (69,95%) | 131 (30,05%) |

Tabel tersebut menunjukkan bahwa distribusi kelas pada data training dan validation tetap terjaga dan sangat mendekati distribusi data keseluruhan. Hal ini menandakan bahwa teknik stratified split berhasil mempertahankan proporsi kelas, sehingga proses pelatihan dan evaluasi menjadi lebih representatif.

Potongan kode pembagian dataset yang digunakan adalah sebagai berikut:

```python
labels_df = pd.read_csv(config.LABELS_CSV)
label_mapping = {'NON-SELF-HARM': 0, 'SELF-HARM': 1}

labels_df = labels_df[labels_df['Label Akhir'].isin(label_mapping.keys())].copy()
labels_df['Label'] = labels_df['Label Akhir'].map(label_mapping)

train_df, val_df = train_test_split(
    labels_df,
    test_size=0.2,
    random_state=config.SEED,
    stratify=labels_df['Label']
)
```

Setelah pembagian dataset selesai, masing-masing subset kemudian dibungkus ke dalam `MultimodalDataset` dan dimuat dengan `DataLoader` untuk proses training dan validasi:

```python
train_ds = MultimodalDataset(
    train_df, config.IMAGES_DIR, electra_tokenizer, clip_processor,
    max_len=config.MAX_LEN, is_train=True
)

val_ds = MultimodalDataset(
    val_df, config.IMAGES_DIR, electra_tokenizer, clip_processor,
    max_len=config.MAX_LEN, is_train=False
)
```

Secara keseluruhan, hasil pre-processing pada penelitian ini menunjukkan bahwa data telah dipersiapkan melalui kombinasi antara augmentasi visual, tokenisasi otomatis pada teks, serta pembagian dataset yang terstratifikasi. Pendekatan ini mendukung proses pelatihan model multimodal agar berlangsung lebih stabil dan tetap sesuai dengan format input yang dibutuhkan oleh arsitektur CLIP dan ELECTRA.
