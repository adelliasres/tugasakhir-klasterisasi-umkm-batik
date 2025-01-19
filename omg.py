import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_selection import mutual_info_classif
from scipy.spatial.distance import mahalanobis
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from scipy.spatial.distance import cdist

# Judul aplikasi
st.title("Aplikasi Clustering UMKM Batik")

# Menu sidebar
menu = st.sidebar.selectbox("Pilih Menu", 
                            ('Preprocessing', 'Clustering K-Medoids', 'Evaluasi Hasil Cluster', 'Hasil Cluster', 'Analisa dan Rekomendasi'))

# Konten berdasarkan menu yang dipilih
if menu == 'Preprocessing':
    st.header("Preprocessing Data")
    # Sub-menu untuk preprocessing
    sub_menu = st.sidebar.selectbox("Pilih Tahapan Preprocessing", ["Data Cleaning", "Transformasi", "Normalisasi MinMax"])

    # Tahapan Data Cleaning
    if sub_menu == "Data Cleaning":
        st.subheader("Data Cleaning")
        uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])

        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            st.write("Data yang diunggah:", df)

            if 'df_cleaned' not in st.session_state:
                st.session_state.df_cleaned = df

            # Pilih kolom untuk dihapus
            columns_to_drop = st.multiselect("Pilih kolom yang ingin dihapus:", st.session_state.df_cleaned.columns)
            if st.button("Hapus Kolom"):
                cols_to_drop_valid = [col for col in columns_to_drop if col in st.session_state.df_cleaned.columns]
                if cols_to_drop_valid:
                    st.session_state.df_cleaned = st.session_state.df_cleaned.drop(columns=cols_to_drop_valid)
                    st.write(f"Kolom '{', '.join(cols_to_drop_valid)}' telah dihapus.")
                    st.write(st.session_state.df_cleaned)

            if st.button("Tangani Missing Values"):
                # Mengisi missing values
                st.session_state.df_cleaned = st.session_state.df_cleaned.fillna({
                    col: st.session_state.df_cleaned[col].mean() if st.session_state.df_cleaned[col].dtype != 'object'
                    else st.session_state.df_cleaned[col].mode()[0]
                    for col in st.session_state.df_cleaned.columns})
    
                st.write("Missing values telah diisi:")
                st.write(st.session_state.df_cleaned)
                st.success('Data Cleaning berhasil dilakukan')

    elif sub_menu == "Transformasi":
        st.subheader("Transformasi Data: Label Encoding")
        
        if 'df_cleaned' in st.session_state:
            df_clean = st.session_state.df_cleaned
            st.write("Data hasil cleaning:", df_clean)
            
            object_columns = df_clean.select_dtypes(include=['object']).columns.tolist()
            excluded_column = 'nama_usaha'
            if excluded_column in object_columns:
                object_columns.remove(excluded_column)
                st.write(f"Kolom '{excluded_column}' telah dikecualikan dari encoding.")

            if len(object_columns) > 0:
                st.write(f"Kolom bertipe object yang akan di-transformasi: {object_columns}")
                
                for column in object_columns:
                    mapping = {
                        "Ada": 3,
                        "Proses Pengurusan": 2,
                        "Tidak ada": 1
                    }
                    # Lakukan transformasi dengan mapping
                    df_clean[column] = df_clean[column].map(mapping)
                
                st.write("Data setelah Transformasi:")
                st.write(df_clean)
                st.session_state.df_cleaned = df_clean
                
                st.success('Transformasi berhasil!')
            else:
                st.warning("Tidak ada kolom bertipe object untuk di-transformasi.")

        else:
            st.error("Data cleaning belum dilakukan.")

    elif sub_menu == "Normalisasi MinMax":
        st.subheader("Normalisasi Min-Max")

        if 'df_cleaned' in st.session_state:
            df_clean = st.session_state.df_cleaned
            if not df_clean.empty:
                numeric_columns = df_clean.select_dtypes(include=['float64', 'int']).columns.tolist()
                
                if len(numeric_columns) > 0:
                    st.write(f"Kolom yang akan dinormalisasi: {numeric_columns}")

                    scaler = MinMaxScaler()
                    df_normalized = pd.DataFrame(scaler.fit_transform(df_clean[numeric_columns]), columns=numeric_columns)
                    non_numeric_df = df_clean.drop(columns=numeric_columns)
                    df_normalized_final = pd.concat([non_numeric_df.reset_index(drop=True), df_normalized.reset_index(drop=True)], axis=1)

                    st.write("Dataset Setelah Normalisasi Min-Max:", df_normalized_final)
                    st.session_state.df_cleaned = df_normalized_final
                    st.success("Normalisasi berhasil!")

                    csv = df_normalized_final.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV Normalized", data=csv, file_name='normalized_data.csv', mime='text/csv')
                else:
                    st.warning("Tidak ada kolom numerik untuk dinormalisasi.")
            else:
                st.error("Dataframe kosong!")
        else:
            st.error("Data cleaning belum dilakukan.")

elif menu == 'Clustering K-Medoids':

    if 'df_cleaned' in st.session_state:
        df_cleaned = st.session_state.df_cleaned
        numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int']).columns.tolist()

        # Memastikan kolom 'Usaha' ada
        if 'nama_usaha' not in df_cleaned:
            st.error("Kolom 'Usaha' tidak ditemukan dalam data.")
            st.stop()
        
        # Menyimpan kolom 'Usaha' sebelum preprocessing
        usaha_column = df_cleaned['nama_usaha']

        if not numeric_columns:
            st.warning("Data tidak memiliki kolom numerik untuk dilakukan clustering.")
            st.stop()

        if 'data_normalized' not in st.session_state:
            scaler = MinMaxScaler()
            data_normalized = scaler.fit_transform(df_cleaned[numeric_columns])
            st.session_state.data_normalized = data_normalized
        else:
            data_normalized = st.session_state.data_normalized

        if 'final_results' not in st.session_state:
            st.session_state.final_results = None

        st.title("K-Medoids Clustering dengan Berbagai Metode Jarak")
        k_value = st.number_input("Masukkan jumlah cluster (K):", min_value=2, max_value=10, value=3, step=1)

        if st.button("Jalankan Clustering"):
            if k_value == 1:
                            st.write("Dengan K=1, semua data akan menjadi satu cluster.")
                            results = [{
                                "Metric": "Euclidean",
                                "Cluster Labels": np.zeros(len(data_normalized), dtype=int),  # Semua data menjadi satu cluster
                                "Medoids": [1]  # Indeks medoid yang bisa dipilih sembarang, misal 0
                            }]
            else:
                results = []

            # Perhitungan untuk Euclidean dan Manhattan
            for metric_name, metric_function in [
                ("Euclidean", distance_metric(type_metric.EUCLIDEAN)),
                ("Manhattan", distance_metric(type_metric.MANHATTAN))
            ]:
                # Inisialisasi medoids awal
                random.seed(42)
                initial_medoids = random.sample(range(len(data_normalized)), k_value)

                # Jalankan k-medoids
                kmedoids_instance = kmedoids(data_normalized, initial_medoids, metric=metric_function)
                kmedoids_instance.process()
                clusters = kmedoids_instance.get_clusters()

                medoids = kmedoids_instance.get_medoids()
                medoids_adjusted = [medoid +1 for medoid in medoids]


                # Periksa medoids untuk setiap metrik
                st.write(f"Medoids untuk {metric_name }", medoids_adjusted)      

                # Simpan hasil clustering ke dalam tabel
                cluster_labels = np.zeros(len(data_normalized), dtype=int)
                for cluster_idx, cluster in enumerate(clusters):
                    cluster_labels[cluster] = cluster_idx +1

                results.append({
                    "Metric": metric_name,
                    "Cluster Labels": cluster_labels,
                    "Medoids": kmedoids_instance.get_medoids()
                })

            st.session_state.final_results = results
            st.session_state.k_value = k_value 

        if st.session_state.final_results:
            for result in st.session_state.final_results:
                st.subheader(f"Metode Jarak {result['Metric']}")

                # Tampilkan tabel hasil clustering
                df_result = pd.DataFrame(data_normalized, columns=numeric_columns)
                df_result["Cluster"] = result["Cluster Labels"]
                # Menambahkan kolom 'Batik' ke dalam hasil clustering
                df_result["Usaha"] = usaha_column
                st.write(df_result)
                st.session_state.df_clustered = data_normalized

        # Fungsi untuk membuat plot interaktif menggunakan Plotly
        def plot_clusters_plotly(data, labels, title):
            # Reduksi dimensi menggunakan PCA menjadi 2 dimensi
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(data)

            # Buat DataFrame untuk mempermudah visualisasi
            df_plot = pd.DataFrame(reduced_data, columns=["PCA komponen 1", "PCA komponen 2"])
            df_plot["Cluster"] = labels

            # Plot menggunakan Plotly Express
            fig = px.scatter(
                df_plot,
                x="PCA komponen 1",
                y="PCA komponen 2",
                color=df_plot["Cluster"].astype(str),  # Konversi cluster menjadi string untuk kategori
                title=title,
                color_discrete_sequence=px.colors.qualitative.Set2,
                template="plotly_white"
            )
            fig.update_traces(marker=dict(size=10, opacity=0.8))
            fig.update_layout(legend_title="Cluster")
            st.plotly_chart(fig)

        if st.session_state.final_results:
            for result in st.session_state.final_results:
                metric_name = result['Metric']
                cluster_labels = result['Cluster Labels']

                st.subheader(f"Plot Persebaran Cluster {metric_name}")
                # Buat plot interaktif untuk masing-masing metode jarak
                plot_clusters_plotly(st.session_state.data_normalized, cluster_labels, f"Persebaran Cluster dengan {metric_name}")
    

elif menu == 'Evaluasi Hasil Cluster':
    # Fungsi untuk menjalankan evaluasi
    def evaluate_clustering(data, k_values, metrics):
        evaluation_results = []

        for metric_name, metric_function in metrics:
            for k in k_values:
                # Inisialisasi medoids awal
                random.seed(42)
                initial_medoids = random.sample(range(len(data)), k)

                # Jalankan k-medoids
                kmedoids_instance = kmedoids(data, initial_medoids, metric=metric_function)
                kmedoids_instance.process()
                clusters = kmedoids_instance.get_clusters()

                # Buat label cluster untuk semua data
                cluster_labels = np.zeros(len(data), dtype=int)
                for cluster_idx, cluster in enumerate(clusters):
                    cluster_labels[cluster] = cluster_idx

                # Evaluasi menggunakan Silhouette dan Davies-Bouldin
                silhouette = silhouette_score(data, cluster_labels)
                db_index = davies_bouldin_score(data, cluster_labels)

                # Simpan hasil evaluasi
                evaluation_results.append({
                    "Metric": metric_name,
                    "K": k,
                    "Silhouette Score": silhouette,
                    "Davies-Bouldin Index": db_index
                })

        return pd.DataFrame(evaluation_results)

    # Evaluasi clustering untuk K dari 2 hingga K yang dipilih
    if 'final_results' in st.session_state:
        k_values = list(range(2, st.session_state.k_value + 1))
        metrics = [
            ("Euclidean", distance_metric(type_metric.EUCLIDEAN)),
            ("Manhattan", distance_metric(type_metric.MANHATTAN))
        ]

        st.subheader("Evaluasi Cluster Silhouette Score & Davies-Bouldin Index")
        eval_df = evaluate_clustering(st.session_state.data_normalized, k_values, metrics)

        # Menampilkan hasil evaluasi dalam tabel
        st.write(eval_df)

        # Menentukan hasil optimal berdasarkan Silhouette dan DBI
        optimal_silhouette = eval_df.loc[eval_df.groupby("Metric")["Silhouette Score"].idxmax()]
        optimal_dbi = eval_df.loc[eval_df.groupby("Metric")["Davies-Bouldin Index"].idxmin()]

        st.subheader("Hasil Optimal Cluster")
        st.write('Berdasarkan Silhouette dan DBI')
        st.write(optimal_silhouette)

    # Fungsi untuk menampilkan grafik evaluasi hasil clustering
    def plot_evaluation_results(eval_df, metric_name):
        metric_data = eval_df[eval_df["Metric"] == metric_name]

        k_values = metric_data["K"]
        silhouette_scores = metric_data["Silhouette Score"]
        dbi_scores = metric_data["Davies-Bouldin Index"]

        # Membuat grafik
        fig, ax1 = plt.subplots()

        # Plot silhouette scores
        ax1.set_xlabel('Jumlah Cluster (K)')
        ax1.set_ylabel('Silhouette Score', color='tab:blue')
        ax1.plot(k_values, silhouette_scores, label='Silhouette Score', color='tab:blue', marker='o')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')

        # Plot DBI pada axis kedua
        ax2 = ax1.twinx()
        ax2.set_ylabel('Davies-Bouldin Index', color='tab:red')
        ax2.plot(k_values, dbi_scores, label='Davies-Bouldin Index', color='tab:red', marker='s')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='upper right')

        # Menambahkan judul
        plt.title(f'Evaluasi Clustering dengan Jarak ({metric_name})')
        fig.tight_layout()

        # Menampilkan grafik di Streamlit
        st.pyplot(fig)

    # Membuat grafik evaluasi untuk Euclidean
    st.subheader("Grafik Evaluasi dengan Jarak Euclidean")
    plot_evaluation_results(eval_df, "Euclidean")

    # Membuat grafik evaluasi untuk Manhattan
    st.subheader("Grafik Evaluasi dengan Jarak Manhattan")
    plot_evaluation_results(eval_df, "Manhattan")


elif menu == 'Hasil Cluster':
    if 'df_cleaned' in st.session_state:
        df_cleaned = st.session_state.df_cleaned

        # Pastikan kolom 'nama_usaha' ada
        if 'nama_usaha' not in df_cleaned.columns:
            st.error("Kolom 'nama_usaha' tidak ditemukan dalam data.")
            st.stop()

        # Simpan kolom 'nama_usaha' ke dalam session_state
        st.session_state.usaha_column = df_cleaned['nama_usaha']

    def display_cluster_results(data, labels, metric_name):
    # Pastikan 'usaha_column' tersedia di session_state
        if 'usaha_column' not in st.session_state:
            st.error("Kolom 'nama_usaha' tidak tersedia untuk ditampilkan.")
            st.stop()

        # Buat DataFrame hasil clustering
        df_result = pd.DataFrame({
            "Usaha": st.session_state.usaha_column,
            "Cluster": labels
        })

        # Hitung jumlah anggota tiap cluster
        cluster_counts = df_result["Cluster"].value_counts().sort_index()

        # Tampilkan tabel hasil clustering
        st.subheader(f"Hasil Cluster untuk Jarak ({metric_name})")
        st.write("Tabel Hasil Clustering")
        st.write(df_result)

        # Tampilkan jumlah anggota tiap cluster
        st.write("Jumlah Anggota Tiap Cluster")
        st.table(pd.DataFrame({
            "Cluster": cluster_counts.index,
            "Jumlah Anggota": cluster_counts.values
        }))

    # Menampilkan hasil clustering untuk masing-masing metode jarak
    if 'final_results' in st.session_state:
        for result in st.session_state.final_results:
            metric_name = result["Metric"]
            cluster_labels = result["Cluster Labels"]
            display_cluster_results(st.session_state.data_normalized, cluster_labels, metric_name)
    

elif menu == 'Analisa dan Rekomendasi':

    # Fungsi untuk menghitung rata-rata variabel tiap cluster dan menampilkan grafik persentase
    def display_cluster_analysis(df, cluster_labels, usaha_column, metric_name):
        # Membuat DataFrame hasil clustering
        df_result = pd.DataFrame({
            "Usaha": usaha_column,
            "Cluster": cluster_labels
        })
        
        # Menambahkan data numerik dari df
        numeric_columns = df.select_dtypes(include=['float64', 'int']).columns.tolist()
        df_result[numeric_columns] = df[numeric_columns]
        
        # Menghitung rata-rata tiap variabel per cluster
        cluster_means = df_result.groupby('Cluster')[numeric_columns].mean()

        # Menampilkan tabel rata-rata per cluster
        st.subheader(f"Rata-rata Variabel tiap Cluster {metric_name}")
        st.write(cluster_means)
        st.write()

        # Menampilkan heatmap untuk rata-rata tiap variabel
        st.subheader(f"Heatmap {metric_name}")
        plt.figure(figsize=(10, 6))
        sns.heatmap(cluster_means, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Heatmap Rata-rata Variabel per Cluster")
        st.pyplot(plt)
        st.write()

        # Memberikan rekomendasi kebijakan berdasarkan hasil clustering
        st.subheader("Rekomendasi Kebijakan Berdasarkan Hasil Clustering")
        recommendations = generate_recommendations(cluster_means, df_result)
        # Iterasi melalui rekomendasi dan tampilkan kebijakan
        for cluster, policy in recommendations.items():
            st.write(f" Cluster {cluster}: {policy}")
        st.write()
        

    # Fungsi untuk memberikan rekomendasi berdasarkan hasil clustering
    def generate_recommendations(cluster_means, cluster_data):
        recommendations = {}
        
        # Identifikasi cluster dengan nilai kecil
        smallest_omzet_cluster = cluster_means[['omzet','aset']].mean(axis=1).idxmin()
        smallest_naker_cluster = cluster_means['jml_naker'].idxmin()
        smallest_mitra_cluster = cluster_means[['kemitraan','lama_usaha']].mean(axis=1).idxmin()
        smallest_combined_cluster = cluster_means['jml_naker'].idxmin()
        smallest_izin_cluster = None

        if 'surat_izin' in cluster_means.columns:
            smallest_izin_cluster = cluster_means['surat_izin'].idxmin()

        # Rekomendasi berdasarkan kondisi
        for cluster in cluster_means.index:
            if cluster == smallest_omzet_cluster:
                recommendations[cluster] = ("Berikan bantuan modal usaha mikro, seperti program Subsidi UMKM")
            elif cluster == smallest_naker_cluster:
                recommendations[cluster] = ("Berikan pelatihan kepada karyawan untuk meningkatkan kompetensi tenaga kerja dengan pelatihan keterampilan SDM")
            elif cluster == smallest_mitra_cluster:
                recommendations[cluster] = ('Adakan pelatihan untuk meningkatkan kemitraan, seperti pelatihan pembangunan jaringan bisnis dengan usaha besar')
            elif cluster == smallest_combined_cluster:
                recommendations[cluster] = ("Tingkatkan kompetensi tenaga kerja dengan memberikan pelatihan keterampilan")
            elif cluster == smallest_izin_cluster:
                recommendations[cluster] = ("Berikan bantuan pendampingan untuk legalitas usaha")

            # Rekomendasi default untuk cluster lain jika tidak memenuhi kriteria khusus
            if cluster not in recommendations:
                recommendations[cluster] = "Berikan bantuan yang bersifat umum untuk mendukung pengembangan UMKM Batik"

        return recommendations

    # Contoh penggunaan pada hasil clustering
    if 'final_results' in st.session_state:
        for result in st.session_state.final_results:
            metric_name = result["Metric"]
            cluster_labels = result["Cluster Labels"]
            display_cluster_analysis(st.session_state.df_cleaned, cluster_labels, st.session_state.usaha_column, metric_name)
