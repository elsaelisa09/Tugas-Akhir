# 4.4 Hasil Pengembangan Model Multimodal

Pada tahap pengembangan model, arsitektur yang telah dirancang pada Bab III direalisasikan ke dalam bentuk implementasi yang digunakan pada proses eksperimen. Fokus pada subbab ini bukan lagi menjelaskan teori atau rancangan konseptual model, melainkan menjabarkan konfigurasi model yang benar-benar diterapkan pada proyek `multimodal_fusion`, termasuk komponen yang dibekukan (`freeze`), komponen yang dilatih (`trainable`), serta alur representasi fitur dari masing-masing modalitas hingga menghasilkan prediksi kelas.

Berdasarkan implementasi yang tersedia pada folder `models`, penelitian ini menggunakan tiga kelompok model, yaitu model unimodal gambar, model unimodal teks, dan model multimodal. Model unimodal digunakan sebagai baseline untuk melihat kemampuan masing-masing modalitas secara terpisah, sedangkan model multimodal digunakan untuk mengevaluasi pengaruh penggabungan informasi visual dan tekstual terhadap tugas klasifikasi meme self-harm.

Perlu dicatat bahwa pada snapshot kode yang tersedia saat ini, implementasi multimodal simple fusion berada pada file `models/archA.py`, sedangkan implementasi multimodal berbasis transformer berada pada file `models/archB.py`. Dengan demikian, penjelasan pada subbab ini disusun berdasarkan implementasi aktual pada file tersebut, bukan berdasarkan nama atau komentar lama yang mungkin masih tersisa pada beberapa bagian proyek.

## 4.4.1 Konfigurasi Model

Secara umum, seluruh model dalam penelitian ini menggunakan backbone yang sama, yaitu `openai/clip-vit-base-patch32` sebagai encoder visual dan `sentinet/suicidality` sebagai encoder teks berbasis ELECTRA. Pada seluruh varian utama, parameter backbone dibekukan selama pelatihan, sehingga proses optimasi difokuskan pada lapisan tambahan seperti projection layer, modul fusion, transformer fusion, dan classifier. Pendekatan ini dipilih agar representasi pra-latih dari model besar tetap dimanfaatkan, namun biaya pelatihan tetap lebih efisien.

Potongan kode pembekuan encoder yang digunakan pada model adalah sebagai berikut:

```python
if freeze_encoders:
    for p in self.clip.parameters():
        p.requires_grad = False
    for p in self.electra.parameters():
        p.requires_grad = False
```

### a. Model Unimodal Gambar

Model unimodal gambar diimplementasikan pada file `models/archA_imgonly.py`. Pada konfigurasi ini, hanya fitur visual dari CLIP yang digunakan sebagai sumber informasi utama. Citra masukan diproses oleh `CLIPVisionModelWithProjection` untuk menghasilkan embedding visual berdimensi 512. Embedding ini kemudian dinormalisasi menggunakan L2 normalization agar berada pada ruang fitur yang lebih stabil sebelum diteruskan ke classifier.

Meskipun antarmuka model tetap menerima masukan teks agar kompatibel dengan pipeline training yang sama, cabang teks tidak dipakai dalam proses inferensi. Pada implementasi ini, representasi teks digantikan oleh vektor nol (`zero vector`) dengan dimensi 256, lalu dikonkatenasikan dengan representasi gambar untuk menjaga bentuk input classifier tetap konsisten dengan arsitektur multimodal. Dengan demikian, classifier tetap menerima representasi gabungan berdimensi `512 + 256 = 768`, tetapi informasi efektif hanya berasal dari modalitas gambar.

Lapisan `project_text` pada baseline ini memang tetap didefinisikan untuk menjaga konsistensi struktur model, namun seluruh parameternya dibekukan dan tidak digunakan dalam forward pass. Proses pelatihan hanya memperbarui bobot classifier MLP tiga lapis, sehingga model ini berfungsi sebagai image-only baseline yang memanfaatkan representasi visual dari CLIP tanpa fine-tuning backbone.

Potongan kode utamanya adalah sebagai berikut:

```python
if hasattr(self.clip, 'get_image_features'):
    img_output = self.clip.get_image_features(pixel_values)
else:
    img_output = self.clip(pixel_values=pixel_values)

img_proj = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-10)
text_proj = img_proj.new_zeros((img_proj.size(0), self.fusion_text_dim))
fused_rep = torch.cat([img_proj, text_proj], dim=-1)
logits = self.classifier(fused_rep)
```

### b. Model Unimodal Teks

Model unimodal teks diimplementasikan pada file `models/archA_textonly.py`. Pada konfigurasi ini, hanya modalitas teks yang digunakan, sedangkan modalitas gambar dinonaktifkan. Teks masukan diproses terlebih dahulu oleh encoder ELECTRA untuk menghasilkan representasi token-level pada `last_hidden_state`. Selanjutnya, representasi tersebut diringkas menggunakan masked mean pooling agar token padding tidak ikut memengaruhi hasil representasi kalimat.

Setelah proses pooling, representasi teks diproyeksikan dari dimensi hidden ELECTRA ke dimensi fusion yang digunakan pada eksperimen, yaitu 256. Berbeda dengan baseline gambar, baseline teks inilah yang benar-benar menggunakan `project_text` sebagai lapisan trainable tambahan sebelum classifier. Sama seperti model image-only, input classifier tetap disusun dalam format gabungan dua modalitas. Namun pada baseline ini, representasi gambar diganti dengan vektor nol berdimensi 512, sedangkan informasi efektif hanya berasal dari cabang teks.

Dengan konfigurasi ini, proses pelatihan difokuskan pada text projection dan classifier, sedangkan encoder CLIP dan ELECTRA tetap dibekukan. Model ini berfungsi sebagai text-only baseline untuk mengevaluasi kekuatan representasi teks secara independen.

Potongan kode pooling teks yang digunakan adalah sebagai berikut:

```python
txt_out = self.electra(input_ids=input_ids, attention_mask=attention_mask)
last_hidden = txt_out.last_hidden_state

attn = attention_mask.unsqueeze(-1).float()
sum_emb = (last_hidden * attn).sum(dim=1)
sum_mask = attn.sum(dim=1).clamp(min=1e-9)
text_emb = sum_emb / sum_mask

text_proj = self.project_text(text_emb)
img_proj = text_proj.new_zeros((text_proj.size(0), self.img_dim))
fused_rep = torch.cat([img_proj, text_proj], dim=-1)
```

### c. Model Multimodal

Model multimodal pada penelitian ini menggabungkan representasi visual dari CLIP dan representasi tekstual dari ELECTRA. Kedua encoder utama dibekukan selama pelatihan, sehingga pembelajaran difokuskan pada lapisan proyeksi, modul fusion, transformer fusion, positional embedding, dan classifier. Pendekatan ini memungkinkan penelitian memanfaatkan kekuatan model pra-latih tanpa harus melakukan fine-tuning penuh terhadap backbone yang berukuran besar.

#### c.1 Model Multimodal dengan Simple Fusion

Varian multimodal simple fusion diimplementasikan pada file `models/archA.py`. Pada arsitektur ini, fitur visual diperoleh dari encoder CLIP dan kemudian dinormalisasi menggunakan L2 normalization. Jika dimensi fitur gambar tidak sama dengan dimensi fusion yang diinginkan, maka fitur tersebut diproyeksikan melalui `project_image`. Sementara itu, fitur teks diperoleh dari ELECTRA melalui masked mean pooling, kemudian dinormalisasi dengan L2 normalization dan diproyeksikan melalui `project_text`.

Setelah kedua modalitas berada pada representasi yang siap digabungkan, model menerapkan salah satu dari beberapa strategi fusion sederhana, yaitu `concatenate`, `addition`, `multiplication`, `gated_fusion`, `attention_fusion`, dan `bilinear_fusion`. Hasil fusion tersebut kemudian diteruskan ke classifier MLP tiga lapis untuk menghasilkan prediksi kelas biner.

Pada implementasi ini tidak digunakan transformer encoder maupun positional embedding. Oleh karena itu, varian ini merepresentasikan skenario late fusion yang lebih langsung, di mana interaksi antarmodalitas dipelajari terutama melalui mekanisme fusion dan classifier.

Potongan kode pemilihan fusion yang digunakan adalah sebagai berikut:

```python
if self.fusion_method == 'concatenate':
    fused_rep = torch.cat([img_proj, text_proj], dim=-1)
elif self.fusion_method == 'addition':
    fused_rep = img_proj + text_proj
elif self.fusion_method == 'multiplication':
    fused_rep = img_proj * text_proj
elif self.fusion_method == 'gated_fusion':
    gate = torch.sigmoid(self.gate_layer(torch.cat([img_proj, text_proj], dim=-1)))
    fused_rep = gate * img_proj + (1.0 - gate) * text_proj
elif self.fusion_method == 'attention_fusion':
    tokens = torch.stack([img_proj, text_proj], dim=1)
    attn_out, _ = self.fusion_attention(tokens, tokens, tokens)
    fused_rep = attn_out.mean(dim=1)
elif self.fusion_method == 'bilinear_fusion':
    fused_rep = self.bilinear_fusion(img_proj, text_proj)
```

#### c.2 Model Multimodal dengan Transformer Fusion

Varian multimodal berbasis transformer diimplementasikan pada file `models/archB.py`. Pada model ini, fitur visual dari CLIP terlebih dahulu dinormalisasi dan diproyeksikan ke dimensi fusion gambar. Fitur teks dari ELECTRA diproses melalui masked mean pooling, dinormalisasi dengan L2 normalization, lalu diproyeksikan ke dimensi fusion teks. Setelah kedua modalitas berada pada dimensi yang sesuai, model tidak langsung menggabungkannya melalui operasi simple fusion, tetapi menyusunnya menjadi token multimodal.

Dalam implementasi saat ini, token teks dibentuk terlebih dahulu dengan cara menggabungkan representasi teks dengan vektor nol pada sisi gambar, sedangkan token gambar dibentuk dengan cara menggabungkan vektor nol pada sisi teks dengan representasi gambar. Kedua token ini kemudian ditumpuk menjadi urutan token multimodal dan diperkaya dengan `pos_embedding` yang bersifat trainable. Setelah itu, urutan token diproses oleh `TransformerEncoder` dua lapis dengan `nhead=8` dan dimensi feed-forward sebesar empat kali dimensi fusion.

Keluaran transformer kemudian diringkas dengan mengambil token pertama sebagai representasi akhir untuk klasifikasi. Representasi tersebut diteruskan ke classifier MLP tiga lapis untuk menghasilkan prediksi kelas biner. Dengan pendekatan ini, model tidak hanya menggabungkan fitur dari dua modalitas, tetapi juga mempelajari interaksi kontekstual antarmodalitas melalui mekanisme self-attention.

Potongan kode fusion transformer yang digunakan adalah sebagai berikut:

```python
self.pos_embedding = nn.Parameter(torch.randn(1, 2, self.fusion_dim))

encoder_layer = nn.TransformerEncoderLayer(
    d_model=self.fusion_dim,
    nhead=8,
    dim_feedforward=self.fusion_dim * 4,
    dropout=0.1,
    batch_first=True
)
self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
```

```python
text_token = torch.cat([text_proj, torch.zeros_like(img_proj)], dim=-1)
img_token = torch.cat([torch.zeros_like(text_proj), img_proj], dim=-1)

tokens = torch.stack([text_token, img_token], dim=1)
tokens = tokens + self.pos_embedding

fused_tokens = self.fusion_transformer(tokens)
fused_rep = fused_tokens[:, 0, :]
logits = self.classifier(fused_rep)
```

Catatan penting: log eksperimen pada folder `src/wandb` menunjukkan adanya run bernama `mlB-image-text` dan `mlB-text-image`. Namun, pada snapshot kode `models/archB.py` yang tersedia saat ini, urutan token yang secara eksplisit diimplementasikan adalah `text` terlebih dahulu kemudian `image`. Oleh karena itu, jika subbab hasil eksperimen nantinya membahas variasi urutan token atau positional embedding, pembahasan tersebut sebaiknya merujuk pada log eksperimen yang tersimpan, sementara implementasi final pada repositori saat ini dapat dianggap mewakili varian `text-first`.

Secara keseluruhan, konfigurasi model pada penelitian ini dirancang untuk membandingkan performa antara pendekatan unimodal dan multimodal, serta untuk mengevaluasi pengaruh strategi fusion sederhana maupun fusion berbasis transformer dalam klasifikasi meme self-harm.

## 4.4.2 Konfigurasi Hyperparameter

Konfigurasi hyperparameter pada penelitian ini disusun berdasarkan skenario eksperimen yang tercatat pada file konfigurasi dan log pelatihan di folder `src/wandb`. Secara umum, seluruh eksperimen menggunakan backbone yang sama, yaitu CLIP sebagai image encoder dan ELECTRA sebagai text encoder, sementara komponen yang dilatih difokuskan pada lapisan tambahan di atas backbone. Pendekatan ini membuat perbandingan antar skenario menjadi lebih adil karena perubahan performa model lebih banyak dipengaruhi oleh konfigurasi fusion dan hyperparameter pelatihan.

Pada tingkat global, beberapa hyperparameter dasar yang digunakan secara konsisten adalah sebagai berikut.

| Hyperparameter | Nilai Umum |
|---|---|
| CLIP model | `openai/clip-vit-base-patch32` |
| ELECTRA model | `sentinet/suicidality` |
| Jumlah kelas | 2 |
| Panjang token maksimum | 128 |
| Ukuran gambar | 224 |
| Seed | 42 |

Selain itu, proses optimasi dan pelatihan menggunakan konfigurasi berikut:

```python
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=learning_rate,
    weight_decay=1e-2
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=1
)

early_stopper = EarlyStopping(patience=config.PATIENCE, mode='max')
```

Untuk fungsi loss, proyek ini mendukung empat strategi utama, yaitu `none`, `class_weight`, `focal`, dan `class_weight_focal`. Implementasi focal loss menggunakan nilai `gamma = 2.0`. Konfigurasi ini terutama digunakan pada eksperimen multimodal untuk menangani ketidakseimbangan kelas.

Potongan kode konfigurasi loss tersebut adalah sebagai berikut:

```python
model.configure_loss_strategy(
    strategy=config.IMBALANCE_STRATEGY,
    class_weights=class_weights,
    focal_gamma=config.FOCAL_GAMMA
)
```

### a. Konfigurasi Hyperparameter Model Unimodal Gambar

Berdasarkan log eksperimen, model baseline image-only dijalankan dalam lima skenario, yaitu `image-only-1` sampai `image-only-5`. Seluruh eksperimen menggunakan dimensi fitur gambar 512 dan dimensi placeholder teks 256 agar tetap kompatibel dengan classifier.

| Nama Eksperimen | Batch Size | Epoch | Learning Rate | Patience |
|---|---:|---:|---:|---:|
| image-only-1 | 16 | 24 | 1e-4 | 5 |
| image-only-2 | 32 | 24 | 1e-5 | 5 |
| image-only-3 | 16 | 12 | 1e-5 | 3 |
| image-only-4 | 8 | 24 | 1e-4 | 5 |
| image-only-5 | 16 | 24 | 1e-6 | 5 |

Dengan konfigurasi tersebut, fokus eksperimen pada baseline gambar adalah menguji pengaruh variasi `batch size`, `learning rate`, `epoch`, dan `patience` terhadap stabilitas pembelajaran pada modalitas visual.

### b. Konfigurasi Hyperparameter Model Unimodal Teks

Model baseline text-only juga dijalankan dalam lima skenario, yaitu `text-only-1` sampai `text-only-5`. Seluruh eksperimen mempertahankan dimensi placeholder gambar 512 dan dimensi fitur teks 256.

| Nama Eksperimen | Batch Size | Epoch | Learning Rate | Patience |
|---|---:|---:|---:|---:|
| text-only-1 | 16 | 24 | 1e-4 | 5 |
| text-only-2 | 32 | 24 | 1e-5 | 5 |
| text-only-3 | 16 | 12 | 1e-5 | 3 |
| text-only-4 | 8 | 24 | 1e-4 | 5 |
| text-only-5 | 16 | 24 | 1e-6 | 5 |

Seperti pada baseline gambar, variasi hyperparameter pada baseline teks diarahkan untuk melihat pengaruh kombinasi `batch size`, `learning rate`, `epoch`, dan `patience` terhadap kemampuan model memanfaatkan informasi tekstual secara independen.

### c. Konfigurasi Hyperparameter Model Multimodal Simple Fusion

Eksperimen multimodal simple fusion dijalankan pada tiga kelompok skenario, yaitu variasi dimensi fitur, variasi strategi penanganan ketidakseimbangan kelas, dan variasi metode fusion.

#### c.1 Variasi Dimensi Fitur

Eksperimen ini menggunakan konfigurasi umum `batch size = 16`, `epoch = 24`, `learning rate = 1e-4`, dan `patience = 5`. Variasi dilakukan pada kombinasi dimensi fitur teks dan dimensi fitur gambar sebagaimana berikut.

| Nama Eksperimen | Dimensi Teks | Dimensi Gambar |
|---|---:|---:|
| mlA-dim-1 | 768 | 512 |
| mlA-dim-2 | 768 | 256 |
| mlA-dim-3 | 512 | 512 |
| mlA-dim-4 | 512 | 256 |
| mlA-dim-5 | 256 | 512 |
| mlA-dim-6 | 256 | 256 |

Eksperimen ini bertujuan mengevaluasi pengaruh ukuran representasi modalitas terhadap kualitas fusion dan efisiensi model.

#### c.2 Variasi Strategi Ketidakseimbangan Kelas

Setelah kombinasi dimensi ditetapkan, eksperimen berikutnya berfokus pada strategi loss untuk menangani distribusi kelas yang tidak seimbang. Berdasarkan log eksperimen, skenario ini dijalankan dengan konfigurasi umum `batch size = 16`, `epoch = 24`, `learning rate = 1e-4`, `patience = 5`, `fusion_image_dim = 256`, dan `fusion_text_dim = 256`.

| Nama Eksperimen | Strategi Loss |
|---|---|
| mlA-class-imbalance-CW | `class_weight` |
| mlA-class-imbalance-FL | `focal` (`gamma = 2.0`) |
| mlA-class-imbalance-CW-FL | `class_weight_focal` (`gamma = 2.0`) |

Eksperimen ini digunakan untuk melihat pengaruh modifikasi loss terhadap sensitivitas model terhadap kelas minoritas.

#### c.3 Variasi Metode Fusion

Kelompok eksperimen berikutnya menguji beberapa metode fusion sederhana dengan konfigurasi umum `batch size = 16`, `epoch = 24`, `learning rate = 1e-4`, `patience = 5`, `fusion_image_dim = 256`, `fusion_text_dim = 256`, serta `focal loss` dengan `gamma = 2.0`.

| Metode Fusion | Keterangan Implementasi |
|---|---|
| `concatenate` | Menggabungkan fitur gambar dan teks secara langsung dengan operasi konkatenasi |
| `addition` | Menjumlahkan dua vektor fitur berdimensi sama |
| `multiplication` | Mengalikan elemen demi elemen dua vektor fitur berdimensi sama |
| `gated_fusion` | Menggunakan gerbang sigmoid untuk mengatur kontribusi gambar dan teks |
| `attention_fusion` | Membentuk dua token modalitas lalu memadukannya dengan multi-head attention |
| `bilinear_fusion` | Menggunakan lapisan bilinear untuk mempelajari interaksi dua modalitas |

Pada implementasi aktual, metode `addition`, `multiplication`, `gated_fusion`, dan `attention_fusion` mensyaratkan dimensi fitur gambar dan teks yang sama, sedangkan `concatenate` dan `bilinear_fusion` dapat bekerja pada dimensi fitur yang berbeda. Pada log eksperimen yang tersimpan, nama run yang muncul secara eksplisit untuk kelompok ini adalah `mlA-add`, `mlA-multiplication`, `mlA-gatedfusion`, `mlA-attnfusion`, dan `mlA-billinearfusion`, sedangkan skenario `concatenate` tercermin sebagai baseline fusion sederhana yang menjadi acuan pada eksperimen multimodal.

### d. Konfigurasi Hyperparameter Model Multimodal Transformer Fusion

Eksperimen multimodal berbasis transformer yang teridentifikasi pada log adalah `mlB-image-text` dan `mlB-text-image`. Keduanya menggunakan konfigurasi umum yang sama, yaitu `batch size = 16`, `epoch = 24`, `learning rate = 1e-4`, `patience = 5`, `fusion_image_dim = 256`, `fusion_text_dim = 256`, dan `focal loss` dengan `gamma = 2.0`.

| Nama Eksperimen | Batch Size | Epoch | Learning Rate | Patience | Dimensi Gambar | Dimensi Teks | Loss |
|---|---:|---:|---:|---:|---:|---:|---|
| mlB-image-text | 16 | 24 | 1e-4 | 5 | 256 | 256 | `focal`, gamma 2.0 |
| mlB-text-image | 16 | 24 | 1e-4 | 5 | 256 | 256 | `focal`, gamma 2.0 |

Secara konseptual, kedua eksperimen tersebut merepresentasikan variasi susunan token multimodal sebelum diproses oleh transformer fusion. Namun, seperti telah dijelaskan pada subbab sebelumnya, snapshot kode yang tersedia saat ini secara eksplisit menunjukkan susunan `text` kemudian `image`. Oleh sebab itu, pembahasan detail mengenai perbedaan performa kedua skenario tersebut lebih tepat ditempatkan pada subbab hasil eksperimen.

Secara keseluruhan, konfigurasi hyperparameter pada penelitian ini dirancang untuk membandingkan baseline unimodal dan model multimodal pada kondisi yang relatif terkontrol. Variasi dilakukan secara bertahap, mulai dari pengujian parameter dasar pada baseline, pengujian dimensi representasi pada multimodal simple fusion, pengujian strategi loss untuk menangani ketidakseimbangan kelas, pengujian beberapa metode fusion, hingga pengujian arsitektur transformer fusion.

## Catatan Penulisan

Untuk struktur laporan, potongan kode yang paling tepat ditempatkan pada Subbab 4.4 adalah potongan yang menjelaskan arsitektur, alur fusion, pembekuan parameter, konfigurasi optimizer, scheduler, dan loss. Sementara itu, potongan kode yang terlalu spesifik pada keluaran eksperimen, seperti logging metrik, confusion matrix, classification report, atau grafik performa, lebih tepat diletakkan pada subbab hasil eksperimen karena bagian tersebut sudah membahas keluaran dan analisis performa model.
