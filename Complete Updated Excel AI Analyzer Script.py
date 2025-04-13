
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class ExcelAIAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Excel AI Analyzer")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        self.data = None
        self.file_path = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="Excel AI Analyzer", 
            font=("Arial", 24, "bold"),
            bg="#f0f0f0"
        )
        title_label.pack(pady=10)
        
        # Description
        desc_label = tk.Label(
            main_frame,
            text="Load Excel files and perform AI analysis on the data",
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        desc_label.pack(pady=5)
        
        # Buttons frame
        button_frame = tk.Frame(main_frame, bg="#f0f0f0")
        button_frame.pack(pady=20)
        
        # Search and Load button
        self.load_button = tk.Button(
            button_frame,
            text="Search and Load",
            command=self.load_excel_file,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="Purple",
            padx=20,
            pady=10
        )
        self.load_button.grid(row=0, column=0, padx=10)
        
        # Apply AI button
        self.ai_button = tk.Button(
            button_frame,
            text="Apply AI",
            command=self.apply_ai_analysis,
            font=("Arial", 12),
            bg="#2196F3",
            fg="Blue",
            padx=20,
            pady=10,
            state=tk.DISABLED  # Initially disabled until a file is loaded
        )
        self.ai_button.grid(row=0, column=1, padx=10)
        
        # Status frame
        status_frame = tk.Frame(main_frame, bg="#f0f0f0")
        status_frame.pack(fill=tk.X, pady=10)
        
        # File status
        self.file_label = tk.Label(
            status_frame,
            text="No file loaded",
            font=("Arial", 10),
            bg="#f0f0f0"
        )
        self.file_label.pack(anchor=tk.W)
        
        # Data preview frame
        preview_frame = tk.LabelFrame(main_frame, text="Data Preview", bg="#f0f0f0", font=("Arial", 12))
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create Treeview for data preview
        self.tree_frame = tk.Frame(preview_frame)
        self.tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.tree_scroll_y = tk.Scrollbar(self.tree_frame)
        self.tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree_scroll_x = tk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL)
        self.tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.tree = ttk.Treeview(
            self.tree_frame,
            yscrollcommand=self.tree_scroll_y.set,
            xscrollcommand=self.tree_scroll_x.set
        )
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        self.tree_scroll_y.config(command=self.tree.yview)
        self.tree_scroll_x.config(command=self.tree.xview)
        
        # Results section
        self.results_frame = tk.LabelFrame(main_frame, text="AI Analysis Results", bg="#f0f0f0", font=("Arial", 12))
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.results_text = tk.Text(self.results_frame, height=10, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def load_excel_file(self):
        """Open a file dialog to select and load an Excel file"""
        filetypes = [
            ("Excel files (*.xlsx)", "*.xlsx"),
            ("Excel files (*.xls)", "*.xls"),
            ("Excel files (*.xlsm)", "*.xlsm"),
            ("Excel files (*.xlsb)", "*.xlsb"),
            ("OpenDocument files", "*.odf *.ods *.odt"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=filetypes
        )
        
        if file_path:
            try:
                if file_path.lower().endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                else:
                    self.data = pd.read_excel(file_path, engine='openpyxl')
                    
                self.file_path = file_path
                filename = os.path.basename(file_path)
                self.file_label.config(text=f"Loaded: {filename}")
                
                # Update the treeview with data
                self.update_data_preview()
                
                # Enable the AI button
                self.ai_button.config(state=tk.NORMAL)
                
                messagebox.showinfo("Success", f"Successfully loaded {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                self.file_label.config(text="Error loading file")
    
    def update_data_preview(self):
        """Update the treeview with data from the loaded Excel file"""
        # Clear existing data
        for i in self.tree.get_children():
            self.tree.delete(i)
            
        # Clear existing columns
        self.tree['columns'] = ()
        
        if self.data is None or self.data.empty:
            return
            
        # Set up columns
        columns = list(self.data.columns)
        self.tree['columns'] = columns
        
        # Configure columns
        self.tree.column("#0", width=0, stretch=tk.NO)
        for col in columns:
            self.tree.column(col, anchor=tk.W, width=100)
            self.tree.heading(col, text=col, anchor=tk.W)
            
        # Add data rows
        for i, row in self.data.head(50).iterrows():  # Show first 50 rows
            values = [row[col] if pd.notna(row[col]) else "" for col in columns]
            self.tree.insert("", tk.END, text=i, values=values)
            
    def apply_ai_analysis(self):
        """Apply AI analysis to the loaded data"""
        if self.data is None:
            messagebox.showerror("Error", "No data loaded. Please load an Excel file first.")
            return
            
        try:
            # Clear previous results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Analyzing data...\n\n")
            self.root.update()
            
            # Perform basic data analysis
            self.perform_data_analysis()
            
            # Perform AI analysis if there are numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                self.perform_ai_analysis(numeric_cols)
            else:
                self.results_text.insert(tk.END, "Not enough numeric columns for AI analysis.\n")
                
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error during analysis: {str(e)}")
            self.results_text.insert(tk.END, f"Error during analysis: {str(e)}\n")
    
    def perform_data_analysis(self):
        """Perform basic data analysis and display results"""
        # Get basic information
        num_rows, num_cols = self.data.shape
        self.results_text.insert(tk.END, f"Dataset contains {num_rows} rows and {num_cols} columns.\n\n")
        
        # Data types
        self.results_text.insert(tk.END, "Column Data Types:\n")
        for col, dtype in self.data.dtypes.items():
            self.results_text.insert(tk.END, f"- {col}: {dtype}\n")
        
        # Missing values
        missing_values = self.data.isnull().sum()
        if missing_values.sum() > 0:
            self.results_text.insert(tk.END, "\nMissing Values:\n")
            for col, count in missing_values.items():
                if count > 0:
                    percentage = (count / num_rows) * 100
                    self.results_text.insert(tk.END, f"- {col}: {count} ({percentage:.2f}%)\n")
        else:
            self.results_text.insert(tk.END, "\nNo missing values found in the dataset.\n")
        
        # Numeric column statistics
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            self.results_text.insert(tk.END, "\nNumeric Column Statistics:\n")
            stats = self.data[numeric_cols].describe().transpose()
            for idx, row in stats.iterrows():
                self.results_text.insert(tk.END, f"- {idx}: Mean={row['mean']:.2f}, Min={row['min']:.2f}, Max={row['max']:.2f}, Std={row['std']:.2f}\n")
        
        # Categorical column information
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            self.results_text.insert(tk.END, "\nCategorical Column Information:\n")
            for col in cat_cols[:5]:  # Limit to first 5 categorical columns
                unique_vals = self.data[col].nunique()
                self.results_text.insert(tk.END, f"- {col}: {unique_vals} unique values\n")
                
                # Show top 3 most common values
                if unique_vals < 100:  # Only for columns with a reasonable number of categories
                    top_vals = self.data[col].value_counts().head(3)
                    self.results_text.insert(tk.END, "  Top values: ")
                    for val, count in top_vals.items():
                        if pd.notna(val):  # Check for NaN values
                            self.results_text.insert(tk.END, f"{val} ({count}), ")
                    self.results_text.insert(tk.END, "\n")
    
    def perform_ai_analysis(self, numeric_cols):
        """Perform AI analysis using scikit-learn"""
        self.results_text.insert(tk.END, "\n--- AI Analysis ---\n\n")
        
        # Prepare data - select only numeric columns and drop rows with missing values
        numeric_data = self.data[numeric_cols].copy()
        numeric_data = numeric_data.dropna()
        
        if len(numeric_data) < 10:
            self.results_text.insert(tk.END, "Not enough complete data rows for AI analysis.\n")
            return
            
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # 1. Correlation Analysis
        self.results_text.insert(tk.END, "Correlation Analysis:\n")
        corr_matrix = numeric_data.corr()
        
        # Find the highest correlations
        corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:  # Only strong correlations
                    corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_value))
        
        if corr_pairs:
            # Sort by absolute correlation value, highest first
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for col1, col2, corr in corr_pairs[:5]:  # Show top 5
                rel_type = "positive" if corr > 0 else "negative"
                self.results_text.insert(tk.END, f"- Strong {rel_type} correlation ({corr:.2f}) between '{col1}' and '{col2}'\n")
        else:
            self.results_text.insert(tk.END, "- No strong correlations found between numeric variables\n")
        
        # 2. Principal Component Analysis (if enough variables)
        if len(numeric_cols) >= 3:
            self.results_text.insert(tk.END, "\nPrincipal Component Analysis:\n")
            pca = PCA()
            pca.fit(scaled_data)
            
            # Explained variance
            exp_var = pca.explained_variance_ratio_
            cum_exp_var = np.cumsum(exp_var)
            
            # Number of components needed to explain 80% of variance
            n_components = np.argmax(cum_exp_var >= 0.8) + 1
            self.results_text.insert(tk.END, f"- {n_components} principal components explain 80% of the data variance\n")
            
            # Top feature contributions to first component
            if len(numeric_cols) > 0:
                top_features = sorted(zip(numeric_cols, abs(pca.components_[0])), key=lambda x: x[1], reverse=True)
                self.results_text.insert(tk.END, "- Most important features in the first principal component:\n")
                for feature, importance in top_features[:3]:  # Top 3
                    self.results_text.insert(tk.END, f"  * {feature} (importance: {importance:.3f})\n")
        
        # 3. Cluster Analysis (if enough data)
        if len(numeric_data) >= 20:
            self.results_text.insert(tk.END, "\nCluster Analysis:\n")
            
            # Determine optimal number of clusters (simplified method)
            max_clusters = min(5, len(numeric_data) // 5)  # Reasonable max number
            if max_clusters >= 2:
                inertia = []
                for n_clusters in range(1, max_clusters + 1):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    kmeans.fit(scaled_data)
                    inertia.append(kmeans.inertia_)
                
                # Simple elbow method (find where inertia starts decreasing more slowly)
                if len(inertia) >= 3:
                    diffs = [inertia[i-1] - inertia[i] for i in range(1, len(inertia))]
                    rel_diffs = [diffs[i] / diffs[i-1] if diffs[i-1] > 0 else 0 for i in range(1, len(diffs))]
                    optimal_clusters = 2  # Default
                    for i, rd in enumerate(rel_diffs):
                        if rd < 0.7:  # If the improvement drops significantly
                            optimal_clusters = i + 2
                            break
                else:
                    optimal_clusters = 2
                
                # Apply K-means with optimal clusters
                kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
                clusters = kmeans.fit_predict(scaled_data)
                
                # Add clusters back to original data for analysis
                cluster_data = numeric_data.copy()
                cluster_data['cluster'] = clusters
                
                # Analyze clusters
                self.results_text.insert(tk.END, f"- Identified {optimal_clusters} natural clusters in the data\n")
                
                for cluster_id in range(optimal_clusters):
                    cluster_size = sum(clusters == cluster_id)
                    cluster_percent = (cluster_size / len(clusters)) * 100
                    self.results_text.insert(tk.END, f"- Cluster {cluster_id+1}: {cluster_size} items ({cluster_percent:.1f}%)\n")
                    
                    # Cluster characteristics (mean values for each feature)
                    cluster_means = cluster_data[cluster_data['cluster'] == cluster_id].mean()
                    overall_means = numeric_data.mean()
                    
                    # Find distinguishing features
                    differences = []
                    for col in numeric_cols:
                        overall_std = numeric_data[col].std()
                        if overall_std > 0:  # Avoid division by zero
                            z_diff = (cluster_means[col] - overall_means[col]) / overall_std
                            differences.append((col, z_diff))
                    
                    # Sort by absolute difference
                    differences.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    # Report top distinguishing features
                    if differences:
                        self.results_text.insert(tk.END, f"  * Distinguished by: ")
                        for feat, z_diff in differences[:2]:  # Top 2 features
                            direction = "higher" if z_diff > 0 else "lower"
                            self.results_text.insert(tk.END, f"{feat} ({direction} by {abs(z_diff):.1f} std), ")
                        self.results_text.insert(tk.END, "\n")
        
        # 4. Summary of findings
        self.results_text.insert(tk.END, "\nSummary of AI Analysis:\n")
        self.results_text.insert(tk.END, "- The data shows ")
        
        if corr_pairs:
            self.results_text.insert(tk.END, f"several strong correlations between variables, particularly between {corr_pairs[0][0]} and {corr_pairs[0][1]}. ")
        else:
            self.results_text.insert(tk.END, "limited correlation between variables. ")
            
        if 'n_components' in locals() and n_components < len(numeric_cols):
            self.results_text.insert(tk.END, f"The dimensionality can be reduced from {len(numeric_cols)} to {n_components} while retaining 80% of information. ")
            
        if 'optimal_clusters' in locals() and optimal_clusters > 1:
            self.results_text.insert(tk.END, f"The data naturally forms into {optimal_clusters} distinct clusters with different characteristics.")
        else:
            self.results_text.insert(tk.END, "The data does not appear to form distinct natural groupings.")

def main():
    root = tk.Tk()
    app = ExcelAIAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
