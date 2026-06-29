"""
RFM + K-Means Clustering
Author : Dhruv Tandel
Applies K-Means to RFM scores. Elbow method validates K=5.
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt, os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

RFM_FILE   = "data/rfm_scores.csv"
OUT_FILE   = "data/rfm_segments.csv"
PLOT_DIR   = "reports"
FEATURES   = ["recency_days","frequency","monetary"]
K_RANGE    = range(2,11)
OPTIMAL_K  = 5
SEED       = 42
SEG_LABELS = {0:"Champions",1:"Loyal Customers",2:"Potential Loyalists",3:"At-Risk",4:"Lost"}

def load_rfm(f):
    df=pd.read_csv(f); print(f"[LOAD] {len(df):,} customers"); return df

def scale(df):
    sc=StandardScaler(); X=sc.fit_transform(df[FEATURES]); return X,sc

def elbow_method(X):
    os.makedirs(PLOT_DIR,exist_ok=True)
    inertias,silhouettes=[],[]
    print("[ELBOW] Testing K =",list(K_RANGE))
    for k in K_RANGE:
        km=KMeans(n_clusters=k,random_state=SEED,n_init=10).fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X,km.labels_,sample_size=min(5000,len(X))))
        print(f"  K={k}  Inertia={km.inertia_:.0f}  Silhouette={silhouettes[-1]:.4f}")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    fig.suptitle("Elbow Method — Optimal K Selection",fontsize=13,fontweight="bold")
    axes[0].plot(list(K_RANGE),inertias,marker="o",color="#4C72B0")
    axes[0].axvline(OPTIMAL_K,color="red",linestyle="--",label=f"K={OPTIMAL_K}")
    axes[0].set(xlabel="K",ylabel="Inertia",title="Inertia vs K"); axes[0].legend()
    axes[1].plot(list(K_RANGE),silhouettes,marker="o",color="#55A868")
    axes[1].axvline(OPTIMAL_K,color="red",linestyle="--",label=f"K={OPTIMAL_K}")
    axes[1].set(xlabel="K",ylabel="Silhouette",title="Silhouette vs K"); axes[1].legend()
    fig.savefig(f"{PLOT_DIR}/elbow_method.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"[ELBOW] Plot saved → {PLOT_DIR}/elbow_method.png")

def fit_kmeans(X,k=OPTIMAL_K):
    km=KMeans(n_clusters=k,random_state=SEED,n_init=10,max_iter=300).fit(X)
    print(f"[KMEANS] K={k}  Inertia={km.inertia_:.0f}"); return km

def assign_labels(df,km,X):
    df=df.copy(); df["cluster"]=km.labels_
    fi={f:i for i,f in enumerate(FEATURES)}
    sorted_c=sorted(range(km.n_clusters),key=lambda c:km.cluster_centers_[c][fi["monetary"]],reverse=True)
    lmap={c:SEG_LABELS[r] for r,c in enumerate(sorted_c)}
    df["segment"]=df["cluster"].map(lmap); return df

def plot_clusters(df,X):
    X2=PCA(n_components=2,random_state=SEED).fit_transform(X)
    colors=["#4C72B0","#55A868","#C44E52","#DD8452","#8172B3"]
    fig,ax=plt.subplots(figsize=(10,7))
    for i,seg in enumerate(sorted(df["segment"].unique())):
        mask=df["segment"]==seg
        ax.scatter(X2[mask,0],X2[mask,1],label=seg,alpha=0.5,s=10,color=colors[i%5])
    ax.set(title="K-Means Segments (PCA 2D)",xlabel="PC1",ylabel="PC2")
    ax.legend(title="Segment",markerscale=3)
    fig.savefig(f"{PLOT_DIR}/kmeans_clusters.png",dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"[PLOT] Cluster chart → {PLOT_DIR}/kmeans_clusters.png")

def run_clustering():
    print("="*55+"\n  K-MEANS CLUSTERING PIPELINE\n"+"="*55)
    df=load_rfm(RFM_FILE); X,_=scale(df)
    print("\n── ELBOW METHOD ──────────────────────────────────")
    elbow_method(X)
    print(f"\n── FITTING K-MEANS (K={OPTIMAL_K}) ──────────────")
    km=fit_kmeans(X); df=assign_labels(df,km,X)
    profile=(df.groupby("segment").agg(customers=("customer_id","count"),
        avg_recency=("recency_days","mean"),avg_monetary=("monetary","mean"),
        total_revenue=("monetary","sum")).round(1).reset_index())
    profile["revenue_pct"]=(profile.total_revenue/profile.total_revenue.sum()*100).round(1)
    print("\n── SEGMENT PROFILE ───────────────────────────────")
    print(profile.sort_values("total_revenue",ascending=False).to_string(index=False))
    plot_clusters(df,X)
    df.to_csv(OUT_FILE,index=False); print(f"\n[SAVE] {OUT_FILE}")

if __name__=="__main__": run_clustering()
