Repository ini berisi tentang cara mengoptimasi waktu inferensi model BERT. Pada kasus ini, model yang menjadi baseline adalah IndoBERT dengan task analisis sentimen.

Ada beberapa cara untuk mengoptimasi waktu inferensi model:
1. Menggunakan model yang sudah terdistilasi (dalam kasus model sejenis BERT, bisa menggunakan model DistilBERT)
2. Melakukan pruning pada model untuk meremove weight dan node yang tidak signifikan.
3. Menyimpan model dalam format ONNX.
4. Mengoptimasi graph untuk model ONNX.
5. Mengkonversi model yang semula berbentuk FP32 menjadi bentuk yang lebih kecil seperti FP16 dan INT8.
6. Menggunakan FastAPI sebagai framework untuk deployment.
7. Menggunakan IOBinding pada FastAPI.
8. Mencari PyTorch threads yang memberikan kecepatan tertinggi.
9. Menggunakan ApacheTVM.
10. Menggunakan OpenVino.