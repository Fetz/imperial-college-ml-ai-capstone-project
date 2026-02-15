import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
import shap
import seaborn as sns


def plot_2D_initial_data(X, y, title, info_text):
    """
    Plot 3d plot & 2d top view to display the known points.
    
    Parameters:
    -----------
    X: input (2D)
    Y: output (1D)
    title : str, plot title
    info_text : text explaining the data
    """

    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(title, fontsize=14)

    # 1. 3D Scatter Plot
    ax_1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax_1.scatter(X[:, 0], X[:, 1], y, c=y,
                     cmap='viridis', s=100, edgecolors='k')
    ax_1.set_xlabel('$x_1$')
    ax_1.set_ylabel('$x_2$')
    ax_1.set_zlabel('$y$')
    ax_1.set_title(f"3D - Visualization")

    # 2. 2D Scatter Plot with Color Mapping
    ax_2 = fig.add_subplot(gs[0, 1])
    scatter_2 = ax_2.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=200, edgecolors='k', zorder=10)

    ax_2.set_xlabel('$x_1$')
    ax_2.set_ylabel('$x_2$')
    ax_2.set_title(f"2D - Top visualisation")
    ax_2.grid(True, linestyle='--', alpha=0.6)
    
    fig.colorbar(scatter_2, ax=ax_2, label='$y$')

    ax_3 = fig.add_subplot(gs[1, :])
    ax_3.axis('off')
    ax_3.text(0.05, 0.95, info_text, fontsize=9, family='monospace',
          horizontalalignment='left',
          verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.show()

def plot_output_outliers(X, y, title, info_text):
    """
    Plot Bar chart to show outliers in the output
    
    Parameters:
    -----------
    X: input (2D)
    Y: output (1D)
    title : str, plot title
    info_text : text explaining the data
    """

    outliers, lower_bound, upper_bound = detect_outliers_iqr(y)

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(title, fontsize=14)

    n_points = len(y)

    ax = fig.add_subplot(gs[0, :])
    colors_bar = ['red' if is_outlier else 'blue' for is_outlier in outliers]
    bars = ax.bar(range(n_points), y, color=colors_bar, edgecolor='black', linewidth=1.5, alpha=0.7)

    # Add horizontal lines for bounds
    ax.axhline(y=lower_bound, color='orange', linestyle='--', linewidth=1.5, 
           label=f'Lower Bound ({lower_bound:.1f})')
    ax.axhline(y=upper_bound, color='orange', linestyle='--', linewidth=1.5, 
           label=f'Upper Bound ({upper_bound:.1f})')

    # Add a text box with outlier information
    ax_2 = fig.add_subplot(gs[1, 1])
    ax_2.axis('off')

    # Add a text box with outlier information
    outlier_indices = np.where(outliers)[0]
    if len(outlier_indices) > 0:
        outlier_text = 'Detected Outliers:\n'
        for idx in outlier_indices:
            outlier_text += f'({X[idx]}: {y[idx]}\n'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax_2.text(0.02, 0.98, outlier_text.strip(),
                fontsize=10, verticalalignment='top', bbox=props)

    ax_3 = fig.add_subplot(gs[1, 0])
    ax_3.axis('off')
    ax_3.text(0.05, 0.95, info_text, fontsize=9, family='monospace',
          horizontalalignment='left',
          verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
     # Add a text box with outlier information
    for i, (bar, value) in enumerate(zip(bars, y)):
        if outliers[i]:
            height = bar.get_height()
            label = f'{value}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    label,
                    ha='center', va='bottom', fontsize=9, fontweight='bold')


    ax.set_xlabel('Data Point Index', fontsize=11)
    ax.set_ylabel('Output Value', fontsize=11)
    ax.set_title('Output Values: Notice the Outlier', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)




def detect_outliers_iqr(data):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    Returns a boolean array indicating which values are outliers.

    Parameters:
    -----------
    data: output (1D)
    """
    data_array = np.array(data)
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = (data_array < lower_bound) | (data_array > upper_bound)
    return outliers, lower_bound, upper_bound


def plot_2D_mean_uncertainty(X, X1_test, X2_test, mu, sigma, X_excluded=None):
    """
    Plot GP mean prediction and uncertainty as 2D contour maps.

    Parameters:
    -----------
    X: training input points used by the model (n_samples, 2)
    X1_test: Prediction grid Dimension 1
    X2_test: Prediction grid Dimension 2
    mu: predicted mean on the grid
    sigma: predicted std on the grid
    X_excluded: optional input points not used by the model (n_samples, 2),
                shown as red triangles to indicate excluded evaluations
    """

    mu_grid = mu.reshape(X1_test.shape)
    sigma_grid = sigma.reshape(X1_test.shape)

    best_idx = np.argmax(mu)
    best_x1 = X1_test.ravel()[best_idx]
    best_x2 = X2_test.ravel()[best_idx]

    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax in (ax_1, ax_2):
        ax.scatter(X[:, 0], X[:, 1], c='blue', s=100,
                edgecolors='black', linewidth=1.5, zorder=5, label='Training points')
        if X_excluded is not None and len(X_excluded) > 0:
            ax.scatter(X_excluded[:, 0], X_excluded[:, 1], c='red', s=100,
                    marker='^', edgecolors='black', linewidth=1.5, zorder=5,
                    label='Excluded points')

    # 2D Mean
    contour = ax_1.contourf(X1_test, X2_test, mu_grid, levels=20, cmap='RdYlGn')
    ax_1.scatter(best_x1, best_x2, c='magenta', s=300, marker='*',
            edgecolors='black', linewidth=1.5, zorder=10, label='Predicted peak')
    ax_1.set_xlabel('Input 1')
    ax_1.set_ylabel('Input 2')
    ax_1.set_title('GP Mean Prediction (Function Shape)')
    ax_1.legend()
    plt.colorbar(contour, ax=ax_1, label='Predicted Output')

    # 2D Uncertainty
    contour = ax_2.contourf(X1_test, X2_test, sigma_grid, levels=20, cmap='YlOrRd')
    ax_2.set_xlabel('Input 1')
    ax_2.set_ylabel('Input 2')
    ax_2.set_title('GP Uncertainty (Where is GP Confident?)')
    plt.colorbar(contour, ax=ax_2, label='Uncertainty (Std Dev)')

    plt.tight_layout()
    plt.show()


def plot_3D_initial_data(X, y, title, info_text):
    """
    Visualize 3D input data points (x1, x2, x3) with a 4th dimension (y) 
    represented by color.
    
    Parameters:
    -----------
    X : ndarray, shape (n_samples, 3) -> The 3D input points
    y : ndarray, shape (n_samples,)    -> The blackbox output
    title : str, plot title
    info_text : str, explanatory text
    """
    
    fig = plt.figure(figsize=(20, 14))
    # 3 rows: Main 3D plot, then Projections, then Info box
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(title, fontsize=18, fontweight='bold')

    # --- 1. Main 3D Scatter (Top Row) ---
    ax_main = fig.add_subplot(gs[0, 1], projection='3d')
    main_sc = ax_main.scatter(X[:, 0], X[:, 1], X[:, 2], 
                              c=y, cmap='viridis', s=100, edgecolors='k', alpha=0.7)
    ax_main.set_xlabel('$x_1$')
    ax_main.set_ylabel('$x_2$')
    ax_main.set_zlabel('$x_3$')
    ax_main.set_title("3D Input Space (Color = $y$)", fontsize=14)
    fig.colorbar(main_sc, ax=ax_main, label='$y$', pad=0.1)

    # --- 2. Projections (Middle Row) ---
    # Projection x1 vs x2
    ax_p1 = fig.add_subplot(gs[1, 0])
    ax_p1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=60, edgecolors='k')
    ax_p1.set_xlabel('$x_1$')
    ax_p1.set_ylabel('$x_2$')
    ax_p1.set_title("Top View ($x_1, x_2$)")
    ax_p1.grid(True, alpha=0.3)

    # Projection x1 vs x3
    ax_p2 = fig.add_subplot(gs[1, 1])
    ax_p2.scatter(X[:, 0], X[:, 2], c=y, cmap='viridis', s=60, edgecolors='k')
    ax_p2.set_xlabel('$x_1$')
    ax_p2.set_ylabel('$x_3$')
    ax_p2.set_title("Side View ($x_1, x_3$)")
    ax_p2.grid(True, alpha=0.3)

    # Projection x2 vs x3
    ax_p3 = fig.add_subplot(gs[1, 2])
    ax_p3.scatter(X[:, 1], X[:, 2], c=y, cmap='viridis', s=60, edgecolors='k')
    ax_p3.set_xlabel('$x_2$')
    ax_p3.set_ylabel('$x_3$')
    ax_p3.set_title("Front View ($x_2, x_3$)")
    ax_p3.grid(True, alpha=0.3)

    # --- 3. Info Text (Bottom Row) ---
    ax_info = fig.add_subplot(gs[0, 0])
    ax_info.axis('off')
    ax_info.text(0.01, 0.5, info_text, fontsize=10, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.show()


def plot_4D_initial_data(X, y, title, info_text):
    """
    Visualize 4D input data points.
    X: [x1, x2, x3, x4]
    y: Output value
    
    Mapping:
    - x, y, z axes -> x1, x2, x3
    - Color        -> x4 (The 4th dimension)
    - Size         -> y  (The output/result)

    Parameters:
    -----------
    X : ndarray, shape (n_samples, 5) -> The 5D input points
    y : ndarray, shape (n_samples,)    -> The blackbox output
    title : str, plot title
    info_text : str, explanatory text
    """

    # Create a single DataFrame
    feature_names = [f'x{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['y (Output)'] = y
    
    # --- 1. Correlation Matrix ---
    # With only 15 points, use Spearman (Rank) correlation 
    # to find monotonic relationships, not just linear ones.
    plt.figure(figsize=(8, 6))
    corr = df.corr(method='spearman')
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap (Spearman)")
    plt.show()

    # --- 2. Pairplot (The "Truth" View) ---
    # This plots every dimension against every other dimension.
    # We color the points by the Output value to see clusters.
    print("Generating Pairplot...")
    g = sns.pairplot(
        df, 
        hue='y (Output)',      # Color points by output value
        palette='viridis',     # Blue=Low, Yellow=High
        diag_kind='kde',       # Show distribution on diagonal
        corner=True,           # Remove redundant top-right plots
        plot_kws={'s': 100, 'alpha': 0.8} # Make points big and visible
    )
    g.figure.suptitle("Pairwise Relationships (Color = Output Magnitude)", y=1.02)
    plt.show()

    # --- 3. Parallel Coordinates (Improved) ---
    # We normalize manually to ensure 0-1 scaling for visualization
    df_norm = (df - df.min()) / (df.max() - df.min())
    
    plt.figure(figsize=(15, 6))
    pd.plotting.parallel_coordinates(
        df_norm.assign(temp_class=pd.qcut(df['y (Output)'], q=3, labels=["Low", "Med", "High"])), 
        'temp_class', 
        color=('#440154', '#21918c', '#fde725'), # Viridis colors
        alpha=0.6
    )
    plt.title("Parallel Coordinates: How Inputs flow to Output Categories")
    plt.grid(alpha=0.3)
    plt.show()


def plot_5D_analysis(X, y, title, info_text):
    """
    Comprehensive 5D Analysis Dashboard.
    Merges Parallel Coordinates (visual flow) with Statistical Analysis (drivers).

    Parameters:
    -----------
    X : ndarray, shape (n_samples, 5) -> The 5D input points
    y : ndarray, shape (n_samples,)    -> The blackbox output
    title : str, plot title
    info_text : str, explanatory text
    """
    
    # --- 1. Prepare Data ---
    n_samples, n_dims = X.shape
    cols = [f'$x_{i+1}$' for i in range(n_dims)]
    df = pd.DataFrame(X, columns=cols)
    df['Target ($y$)'] = y
    
    # Normalize for Parallel Coordinates (0-1 scale)
    df_norm = (df - df.min()) / (df.max() - df.min())
    
    # Calculate Spearman Correlation (Best for Black Box/Non-linear)
    corr_matrix = df.corr(method='spearman')

    # --- 2. Setup Figure Layout ---
    fig = plt.figure(figsize=(20, 12))
    # Grid: Top row (Parallel Coords), Bottom row split (Heatmap | Bar Chart)
    gs = GridSpec(3, 2, height_ratios=[1.2, 1.5, .8], figure=fig, hspace=0.3, wspace=0.2)
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.96)

    # --- 3. Top: Parallel Coordinates ---
    ax_pc = fig.add_subplot(gs[0, :])
    
    # Use a diverging colormap (Blue=Low y, Red=High y) to make patterns obvious
    cmap = plt.get_cmap('Spectral_r') 
    
    # Plot lines
    for i in range(len(df_norm)):
        # Get normalized y value for color
        y_val_norm = df_norm.iloc[i]['Target ($y$)']
        ax_pc.plot(range(n_dims), df_norm.iloc[i][:-1], # Exclude y from the line plot x-axis
                   color=cmap(y_val_norm), 
                   alpha=0.6, linewidth=2)
    
    ax_pc.set_xticks(range(n_dims))
    ax_pc.set_xticklabels(cols, fontsize=12)
    ax_pc.set_title("Parallel Coordinates: Trace how inputs flow to Output (Red = High Output)", fontsize=14)
    ax_pc.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Add Colorbar for PC
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=y.min(), vmax=y.max()))
    cbar = fig.colorbar(sm, ax=ax_pc, pad=0.01)
    cbar.set_label('Output ($y$)', rotation=270, labelpad=15)

    # --- 4. Bottom Left: Correlation Heatmap ---
    ax_corr = fig.add_subplot(gs[1, 0])
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=ax_corr, fmt=".2f", vmin=-1, vmax=1,
                cbar_kws={'label': 'Spearman Correlation'})
    ax_corr.set_title("Correlation Matrix (Red = Positive Driver, Blue = Negative)", fontsize=14)

    # --- 5. Bottom Right: Feature Importance ---
    ax_imp = fig.add_subplot(gs[1, 1])
    
    # Extract correlation of inputs with Target
    importance = corr_matrix['Target ($y$)'].drop('Target ($y$)')
    
    # Color bars by sign (Red for positive correlation, Blue for negative)
    # This is better than absolute value because it tells you Direction too.
    bar_colors = ['#d73027' if v >= 0 else '#4575b4' for v in importance]
    
    importance.plot(kind='barh', color=bar_colors, ax=ax_imp, edgecolor='black', alpha=0.8)
    
    ax_imp.set_title("Feature Influence: Which inputs drive $y$?", fontsize=14)
    ax_imp.set_xlabel("Correlation Coefficient (Length = Strength)", fontsize=12)
    ax_imp.axvline(0, color='black', linewidth=1) # Zero line
    ax_imp.grid(axis='x', linestyle='--', alpha=0.5)
    
    # --- 6. Info Box ---
    # Add info text in a corner
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.axis('off')
    ax_info.text(0.02, 0.02, info_text, fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.8))

    plt.show()

def plot_6D_blackbox_analysis(X, y, title, info_text):
    """
    Analyzes 6D data using Machine Learning to interpret the blackbox.
    X: [x1, x2, x3, x4, x5, x6]
    y: Output value
    """
    # 1. Train a Constrained Random Forest
    # We restrict max_depth=3 to prevent overfitting on such small data.
    model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X, y)
    
    feature_names = [f'$x_{i+1}$' for i in range(X.shape[1])]
    
    # 2. Setup Figure Layout
    fig = plt.figure(figsize=(20, 16))
    # Grid: 
    # Row 0: Parallel Coordinates (Raw Data)
    # Row 1: Feature Importance + Correlation Heatmap (Statistics)
    # Row 2: Partial Dependence Plots (Model Shape)
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)

    # --- ROW 1: REALITY CHECK (Parallel Coordinates) ---
    ax_pc = fig.add_subplot(gs[0, :])
    
    # Normalize data for visualization
    df = pd.DataFrame(X, columns=feature_names)
    df_norm = (df - df.min()) / (df.max() - df.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    
    # Plot lines colored by Output
    cmap = plt.get_cmap('viridis')
    for i in range(len(df_norm)):
        ax_pc.plot(range(6), df_norm.iloc[i], color=cmap(y_norm[i]), alpha=0.6, linewidth=2)
    
    ax_pc.set_xticks(range(6))
    ax_pc.set_xticklabels(feature_names, fontsize=12)
    ax_pc.set_title("1. Raw Data Reality: Parallel Coordinates (Yellow = High Output)", fontsize=14)
    ax_pc.grid(alpha=0.3)
    
    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=y.min(), vmax=y.max()))
    cbar = fig.colorbar(sm, ax=ax_pc, pad=0.01)
    cbar.set_label('Output ($y$)', rotation=270, labelpad=15)


    # --- ROW 2: STATISTICS (Importance & Correlation) ---
    
    # A. Feature Importance (Bar Chart)
    ax_imp = fig.add_subplot(gs[1, :2]) # Spans left half
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    # Color bars: Green if significant (> 0.1), Gray if noise
    colors = ['teal' if imp > 0.1 else 'silver' for imp in importances[indices]]
    
    ax_imp.barh(range(len(indices)), importances[indices], color=colors, align='center')
    ax_imp.set_yticks(range(len(indices)))
    ax_imp.set_yticklabels([feature_names[i] for i in indices])
    ax_imp.set_title("2a. Random Forest Feature Importance", fontsize=14)
    ax_imp.set_xlabel("Contribution to Variance (Gray = Likely Noise)")

    # B. Correlation Matrix (Heatmap)
    ax_corr = fig.add_subplot(gs[1, 2:]) # Spans right half
    df['y'] = y
    corr = df.corr(method='spearman')
    # Mask the diagonal
    mask = np.eye(len(corr), dtype=bool)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                vmin=-1, vmax=1, ax=ax_corr, cbar=False)
    ax_corr.set_title("2b. Spearman Correlation (Linear Verification)", fontsize=14)


    # --- ROW 3: MODEL SHAPE (Partial Dependence) ---
    # We plot the Top 4 features
    top_features = indices[-4:][::-1] # Get top 4, reversed (highest first)
    
    for i, feat_idx in enumerate(top_features):
        ax_pdp = fig.add_subplot(gs[2, i])
        
        # Partial Dependence Plot
        PartialDependenceDisplay.from_estimator(
            model, X, [feat_idx], feature_names=feature_names, ax=ax_pdp,
            line_kw={"color": "crimson", "linewidth": 3}
        )
        
        # Add a rug plot (little ticks) to show where we actually have data
        # This is CRITICAL for small data so we don't trust empty regions
        ax_pdp.plot(X[:, feat_idx], np.full(len(X), ax_pdp.get_ylim()[0]), 
                   '|', color='k', markersize=10, label='Data Points')
        
        ax_pdp.set_title(f"Effect of {feature_names[feat_idx]}", fontsize=12)
        ax_pdp.grid(True, alpha=0.3)

    plt.show()

def plot_8D_shap_analysis(X, y):
    """
    Interpretation of an 8D function using SHAP values

    Parameters:
    -----------
    X : ndarray, shape (n_samples, 8) -> The 8D input points
    y : ndarray, shape (n_samples,)    -> The blackbox output
    """
    feature_names = [f'$x_{i+1}$' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    # 1. Train a Constrained Model
    # We limit depth to 4 to prevent the model from memorizing the 15 points.
    model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_df, y)

    # 2. Compute SHAP Values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # 3. Setup Visualization Grid
    fig = plt.figure(figsize=(24, 14))
    # Grid: Left side (SHAP Beeswarm), Right side (Correlation + Top Feature)
    gs = GridSpec(2, 2, height_ratios=[1.3, 1], figure=fig, hspace=.5, wspace=.3)

    # --- LEFT: SHAP Summary Plot (The "Beeswarm") ---
    # This plot combines Feature Importance with Feature Effect.
    ax_shap = fig.add_subplot(gs[:, 0]) # Spans both rows on the left
    
    # We plot inside the subplot. show=False prevents it from closing the figure.
    plt.sca(ax_shap) 
    shap.summary_plot(shap_values, X_df, plot_type="dot", show=False, feature_names=feature_names)
    ax_shap.set_title("How each variable pushes the prediction", fontsize=10)
    
    # --- RIGHT TOP: Correlation Heatmap (The Sanity Check) ---
    ax_corr = fig.add_subplot(gs[0, 1])
    
    # Calculate Spearman Correlation (Robust to outliers)
    X_df['$y$'] = y
    corr = X_df.corr(method='spearman')
    
    # Mask the diagonal and irrelevant (low) correlations to reduce noise
    mask = np.eye(len(corr), dtype=bool) | (np.abs(corr) < 0.3) 
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', 
                vmin=-1, vmax=1, ax=ax_corr, cbar_kws={'label': 'Correlation'})
    ax_corr.set_title("Significant Linear Correlations", fontsize=10)

    # --- RIGHT BOTTOM: Dependence Plot (Top Driver) ---
    ax_dep = fig.add_subplot(gs[1, 1])
    
    # Identify the most important feature (by mean absolute SHAP value)
    feature_importance = np.abs(shap_values).mean(0)
    top_feature_idx = np.argsort(feature_importance)[-1] # Last one is biggest
    top_feature_name = feature_names[top_feature_idx]
    
    # Plot Feature Value (x-axis) vs SHAP Value (y-axis)
    # This reveals the "Shape" of the function (Linear, U-shape, Threshold)
    ax_dep.scatter(X[:, top_feature_idx], shap_values[:, top_feature_idx], 
                   c=y, cmap='viridis', s=100, edgecolors='k')
    
    ax_dep.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_dep.set_xlabel(f"Value of {top_feature_name}")
    ax_dep.set_ylabel(f"Impact SHAP)")
    ax_dep.set_title(f"3. Shape of Top Driver ({top_feature_name})", fontsize=10)
    ax_dep.grid(True, alpha=0.3)

    plt.show()

def plot_3D_mean_uncertainty(X, model, X1_test, X2_test, x3_slice_value=0.5):
    """
    Visualizes a 3D GP by taking a 2D slice at a specific X3 value.
    """
    # 1. Create the prediction mesh for the slice
    # Assuming X1_test and X2_test are 2D arrays from np.meshgrid
    x1_flat = X1_test.ravel()
    x2_flat = X2_test.ravel()
    x3_flat = np.full_like(x1_flat, x3_slice_value)
    
    X_slice = np.vstack([x1_flat, x2_flat, x3_flat]).T
    
    # 2. Get GP Predictions
    mu, sigma = model.predict(X_slice, return_std=True)
    
    mu_grid = mu.reshape(X1_test.shape)
    sigma_grid = sigma.reshape(X1_test.shape)

    # 3. Plotting
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2)
    
    # Filter training points near the slice for context (optional but helpful)
    # Points within 0.1 range of the slice value
    mask = np.abs(X[:, 2] - x3_slice_value) < 0.1
    X_near = X[mask]

    # Plot 1: Mean Contour
    ax1 = fig.add_subplot(gs[0, 0])
    cf1 = ax1.contourf(X1_test, X2_test, mu_grid, levels=20, cmap='RdYlGn')
    ax1.scatter(X_near[:, 0], X_near[:, 1], c='blue', edgecolors='k', label='Points near slice')
    ax1.set_title(f'Mean Prediction (Slice X3={x3_slice_value})')
    plt.colorbar(cf1, ax=ax1)

    # Plot 2: Uncertainty Contour
    ax2 = fig.add_subplot(gs[0, 1])
    cf2 = ax2.contourf(X1_test, X2_test, sigma_grid, levels=20, cmap='YlOrRd')
    ax2.set_title('Uncertainty (Std Dev)')
    plt.colorbar(cf2, ax=ax2)

    # Plot 3: 3D Surface Mean
    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    ax3.plot_surface(X1_test, X2_test, mu_grid, cmap='RdYlGn', alpha=0.8)
    ax3.set_zlabel('Output')
    ax3.set_title('3D Surface Mean')

    # Plot 4: 3D Surface Uncertainty
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    ax4.plot_surface(X1_test, X2_test, sigma_grid, cmap='YlOrRd', alpha=0.8)
    ax4.set_zlabel('Sigma')
    ax4.set_title('3D Surface Uncertainty')

    plt.show()


def plot_bar(raw, values, y_label, title):
    x = np.arange(len(raw))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, values, color='skyblue')

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{float(v) - 10.1:.12f}' for v in raw], rotation=45)

    plt.tight_layout()


def plot_bar_diff(raw, before, after, label_before, label_after, y_label, title):
    width = 0.35
    x = np.arange(len(raw))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width/2, before, width, label=label_before, color='lightgrey')
    ax.bar(x + width/2, after, width, label=label_after, color='skyblue')

    # Labeling and Formatting
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{float(v) - 10.1:.12f}' for v in raw], rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_log_transform(y_raw, y_log, pos_mask, title='Log10 Transform (Positive Points)', show_top_n=None):
    """
    Bar chart showing log10 values for positive points, with placeholders for excluded (negative) points.

    Parameters:
    -----------
    y_raw : ndarray, raw output values
    y_log : ndarray, log10 values of positive points only
    pos_mask : ndarray, boolean mask for positive points
    title : str, plot title
    show_top_n : int or None, if set show only the N smallest and N largest outputs (by y_raw value)
    """
    n_points = len(y_raw)

    # Build full-length arrays for plotting
    log_vals = np.full(n_points, np.nan)
    log_vals[pos_mask] = y_log

    # Determine which indices to display
    if show_top_n is not None and show_top_n < n_points // 2:
        sorted_indices = np.argsort(y_raw)
        bottom_indices = sorted_indices[:show_top_n]
        top_indices = sorted_indices[-show_top_n:]
        display_indices = np.sort(np.concatenate([bottom_indices, top_indices]))
        display_label = f'Showing {show_top_n} smallest + {show_top_n} largest of {n_points} points'
    else:
        display_indices = np.arange(n_points)
        display_label = None

    n_display = len(display_indices)
    x = np.arange(n_display)

    fig_width = max(14, n_display * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    # Map display positions to original indices
    disp_log_vals = log_vals[display_indices]
    disp_pos_mask = pos_mask[display_indices]
    disp_neg_mask = ~disp_pos_mask

    # Light red placeholder bars for excluded points
    if disp_neg_mask.any():
        excluded_heights = np.where(disp_neg_mask, y_log.min() * 1.05, np.nan)
        ax.bar(x[disp_neg_mask], excluded_heights[disp_neg_mask], color='lightcoral', edgecolor='black',
               alpha=0.4, label=f'Excluded (negative): {(~pos_mask).sum()} pts')

    # Blue bars for included log10 values
    if disp_pos_mask.any():
        ax.bar(x[disp_pos_mask], disp_log_vals[disp_pos_mask], color='steelblue', edgecolor='black',
               alpha=0.8, label=f'Included (positive): {pos_mask.sum()} pts')

    # Annotate each bar with its log10 value
    fontsize = 8 if n_display <= 22 else 7
    for j in np.where(disp_pos_mask)[0]:
        val = disp_log_vals[j]
        ax.text(j, val + 0.02 * abs(val), f'{val:.1f}', ha='center', va='bottom', fontsize=fontsize)

    # Pad y-axis so labels don't overflow
    visible_vals = disp_log_vals[~np.isnan(disp_log_vals)]
    if len(visible_vals) > 0:
        y_range = visible_vals.max() - visible_vals.min()
        padding = max(0.15 * y_range, 0.5)
        ax.set_ylim(visible_vals.min() - 0.1 * y_range, visible_vals.max() + padding)

    ax.set_xlabel('Data Point Index', fontsize=11)
    ax.set_ylabel('log10(y)', fontsize=11)
    full_title = title
    if display_label:
        full_title += f'\n({display_label})'
    ax.set_title(full_title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_indices, fontsize=fontsize)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_svm_analysis(X1_test, X2_test, svm_proba, mu_svr_log, mu_svr_qt,
                      X_train, X_train_pos, X_train_neg, svm_labels):
    """
    3-panel plot: SVM classifier probability, SVR log-space surrogate, SVR QT surrogate.

    Parameters:
    -----------
    X1_test, X2_test : meshgrid arrays for the prediction surface
    svm_proba : SVM classifier P(promising) on the grid
    mu_svr_log : SVR predictions in log-space on the grid
    mu_svr_qt : SVR predictions in QT-space on the grid
    X_train : all training points (scaled)
    X_train_pos : positive training points (scaled)
    X_train_neg : negative training points (scaled)
    svm_labels : binary labels (1=promising, 0=not)
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle("SVM Analysis", fontsize=14)

    # SVM Classifier
    ax = axes[0]
    cf = ax.contourf(X1_test, X2_test, svm_proba.reshape(X1_test.shape), levels=20, cmap='RdYlGn')
    ax.scatter(X_train[svm_labels==1, 0], X_train[svm_labels==1, 1],
               c='green', s=150, edgecolors='k', label='Promising', zorder=5)
    ax.scatter(X_train[svm_labels==0, 0], X_train[svm_labels==0, 1],
               c='red', s=150, edgecolors='k', label='Not promising', zorder=5)
    ax.set_xlabel('$x_1$ (scaled)')
    ax.set_ylabel('$x_2$ (scaled)')
    ax.set_title('SVM Classifier: P(promising region)')
    ax.legend()
    plt.colorbar(cf, ax=ax)

    # SVR Surrogate on log-space (positive only)
    ax = axes[1]
    cf = ax.contourf(X1_test, X2_test, mu_svr_log.reshape(X1_test.shape), levels=20, cmap='RdYlGn')
    ax.scatter(X_train_pos[:, 0], X_train_pos[:, 1], c='blue', s=150, edgecolors='k', zorder=5)
    ax.scatter(X_train_neg[:, 0], X_train_neg[:, 1], c='red', s=150, marker='^', edgecolors='k', zorder=5, label='Excluded')
    ax.set_xlabel('$x_1$ (scaled)')
    ax.set_ylabel('$x_2$ (scaled)')
    ax.set_title('SVR Surrogate: log10(y) [pos only]')
    ax.legend()
    plt.colorbar(cf, ax=ax, label='log10(y)')

    # SVR Surrogate on QuantileTransformer (all points)
    ax = axes[2]
    cf = ax.contourf(X1_test, X2_test, mu_svr_qt.reshape(X1_test.shape), levels=20, cmap='RdYlGn')
    ax.scatter(X_train[:, 0], X_train[:, 1], c='blue', s=150, edgecolors='k', zorder=5)
    ax.set_xlabel('$x_1$ (scaled)')
    ax.set_ylabel('$x_2$ (scaled)')
    ax.set_title('SVR Surrogate: y_qt [all points]')
    plt.colorbar(cf, ax=ax, label='QuantileTransformer(y)')

    plt.tight_layout()
    plt.show()


def plot_acquisition_comparison(X1_test, X2_test, surrogates_dict, svm_proba,
                                ensemble_ucb, X_train_pos, X_train_neg,
                                best_points, ensemble_best_norm):
    """
    6-panel plot: 4 surrogate constrained UCBs + SVM probability + ensemble UCB.

    Parameters:
    -----------
    X1_test, X2_test : meshgrid arrays for the prediction surface
    surrogates_dict : dict mapping name -> constrained UCB values on the grid
    svm_proba : SVM classifier P(promising) on the grid
    ensemble_ucb : ensemble-averaged UCB on the grid
    X_train_pos : positive training points (scaled)
    X_train_neg : negative training points (scaled)
    best_points : dict mapping name -> {'norm': array, ...} for each surrogate's best point
    ensemble_best_norm : best ensemble point in normalised space
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("Acquisition Function Comparison (SVM-constrained UCB)", fontsize=14)

    def _scatter_all(ax):
        ax.scatter(X_train_pos[:, 0], X_train_pos[:, 1], c='blue', s=80,
                   edgecolors='k', zorder=5, label='Pos. training')
        ax.scatter(X_train_neg[:, 0], X_train_neg[:, 1], c='red', s=80,
                   marker='^', edgecolors='k', zorder=5, label='Neg. training')

    for i, (name, ucb_vals) in enumerate(surrogates_dict.items()):
        ax = axes[i // 3, i % 3]
        cf = ax.contourf(X1_test, X2_test, ucb_vals.reshape(X1_test.shape), levels=20, cmap='YlOrRd')
        _scatter_all(ax)
        bp = best_points[name]
        ax.scatter(bp['norm'][0], bp['norm'][1],
                   c='lime', s=300, marker='*', edgecolors='k', zorder=10, label='Suggested')
        ax.set_title(f'{name}')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.legend(fontsize=7)
        plt.colorbar(cf, ax=ax)

    # SVM constraint plot
    ax = axes[1, 1]
    cf = ax.contourf(X1_test, X2_test, svm_proba.reshape(X1_test.shape), levels=20, cmap='RdYlGn')
    _scatter_all(ax)
    ax.set_title('SVM P(promising)')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(fontsize=7)
    plt.colorbar(cf, ax=ax)

    # Ensemble UCB
    ax = axes[1, 2]
    cf = ax.contourf(X1_test, X2_test, ensemble_ucb.reshape(X1_test.shape), levels=20, cmap='YlOrRd')
    _scatter_all(ax)
    ax.scatter(ensemble_best_norm[0], ensemble_best_norm[1],
               c='lime', s=300, marker='*', edgecolors='k', zorder=10, label='ENSEMBLE pick')
    ax.set_title('Ensemble (avg of 4 surrogates)')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend(fontsize=7)
    plt.colorbar(cf, ax=ax)

    plt.tight_layout()
    plt.show()

def plot_3D_mean_uncertainty_slice(X_train, X1, X2, mu, sigma, X_excluded=None, title_prefix='', x3_slice_val=0.5):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{title_prefix} - Slice at X3 = {x3_slice_val:.2f}', fontsize=16)

    # Plot Mean
    contour = axes[0].contourf(X1, X2, mu, 100, cmap='viridis')
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c='r', marker='x', label='Training Points')
    if X_excluded is not None:
        axes[0].scatter(X_excluded[:, 0], X_excluded[:, 1], c='k', marker='o', s=10, label='Excluded Points')
    axes[0].set_title('Mean Prediction')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    fig.colorbar(contour, ax=axes[0])
    axes[0].legend()

    # Plot Uncertainty (Sigma)
    contour = axes[1].contourf(X1, X2, sigma, 100, cmap='inferno')
    axes[1].scatter(X_train[:, 0], X_train[:, 1], c='r', marker='x', label='Training Points')
    if X_excluded is not None:
        axes[1].scatter(X_excluded[:, 0], X_excluded[:, 1], c='k', marker='o', s=10, label='Excluded Points')
    axes[1].set_title('Uncertainty (Std. Dev.)')
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    fig.colorbar(contour, ax=axes[1])
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_svm_analysis_slice(X1, X2, svm_proba, mu_svr_log, mu_svr_qt, X_train, X_train_pos, X_train_neg, svm_labels, x3_slice_val):
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle(f'SVM Analysis - Slice at X3 = {x3_slice_val:.2f}', fontsize=16)
    
    # Plot SVM Classifier Probability
    ax = axes[0]
    contour = ax.contourf(X1, X2, svm_proba, 100, cmap='Greens')
    promising = X_train[svm_labels==1]
    unpromising = X_train[svm_labels==0]
    ax.scatter(promising[:, 0], promising[:, 1], c='blue', marker='o', s=50, label='Promising')
    ax.scatter(unpromising[:, 0], unpromising[:, 1], c='red', marker='x', s=50, label='Not Promising')
    ax.set_title('SVM Classifier P(Promising)')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    fig.colorbar(contour, ax=ax)
    ax.legend()

    # Plot SVR (log-space)
    ax = axes[1]
    contour = ax.contourf(X1, X2, mu_svr_log, 100, cmap='viridis')
    ax.scatter(X_train_pos[:, 0], X_train_pos[:, 1], c='r', marker='x', label='Positive Training Points')
    ax.set_title('SVR Surrogate (log-space)')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    fig.colorbar(contour, ax=ax)
    ax.legend()
    
    # Plot SVR (QuantileTransformer)
    ax = axes[2]
    contour = ax.contourf(X1, X2, mu_svr_qt, 100, cmap='viridis')
    ax.scatter(X_train[:, 0], X_train[:, 1], c='r', marker='x', label='All Training Points')
    ax.set_title('SVR Surrogate (QuantileTransformer)')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    fig.colorbar(contour, ax=ax)
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_acquisition_comparison_slice(X1, X2, surrogates, svm_proba, ensemble_ucb, X_train_pos, X_train_neg, x3_slice_val):
    n_surrogates = len(surrogates)
    # Layout: surrogates on top row, SVM + Ensemble on bottom row
    n_cols = max(n_surrogates, 2)
    fig, axes = plt.subplots(2, n_cols, figsize=(8 * n_cols, 12))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(f'Acquisition Functions (SVM-constrained UCB) - Slice at X3 = {x3_slice_val:.2f}', fontsize=16)

    # Plot individual surrogates (top row)
    for i, (name, ucb_vals) in enumerate(surrogates.items()):
        ax = axes[0, i]
        contour = ax.contourf(X1, X2, ucb_vals, 100, cmap='cividis')
        ax.scatter(X_train_pos[:, 0], X_train_pos[:, 1], c='r', marker='x', s=20, label='Positive Points')
        if X_train_neg.any():
            ax.scatter(X_train_neg[:, 0], X_train_neg[:, 1], c='k', marker='o', s=10, label='Negative Points')

        best_idx_slice = np.unravel_index(np.argmax(ucb_vals), ucb_vals.shape)
        ax.plot(X1[best_idx_slice], X2[best_idx_slice], 'y*', markersize=15, label='Slice Max')

        ax.set_title(name)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        fig.colorbar(contour, ax=ax)
        ax.legend()

    # Hide unused top-row axes
    for i in range(n_surrogates, n_cols):
        axes[0, i].set_visible(False)

    # Plot SVM Probability (bottom left)
    ax = axes[1, 0]
    contour = ax.contourf(X1, X2, svm_proba, 100, cmap='Greens')
    ax.set_title('SVM P(Promising)')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    fig.colorbar(contour, ax=ax)

    # Plot Ensemble UCB (bottom right)
    ax = axes[1, 1]
    contour = ax.contourf(X1, X2, ensemble_ucb, 100, cmap='plasma')
    ax.scatter(X_train_pos[:, 0], X_train_pos[:, 1], c='r', marker='x', s=20, label='Positive Points')
    if X_train_neg.any():
        ax.scatter(X_train_neg[:, 0], X_train_neg[:, 1], c='k', marker='o', s=10, label='Negative Points')

    best_idx_slice = np.unravel_index(np.argmax(ensemble_ucb), ensemble_ucb.shape)
    ax.plot(X1[best_idx_slice], X2[best_idx_slice], 'y*', markersize=15, label='Slice Max')

    ax.set_title('Ensemble UCB (Average of Normalized)')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    fig.colorbar(contour, ax=ax)
    ax.legend()

    # Hide unused bottom-row axes
    for i in range(2, n_cols):
        axes[1, i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_nd_mean_uncertainty_slice(X_train, X1, X2, mu_grid, sigma_grid,
                                   dim1_idx, dim2_idx, X_excluded=None,
                                   title_prefix='', fixed_info=''):
    """
    2-panel contour (mean + uncertainty) for an N-D GP, showing a 2D slice.

    Parameters:
    -----------
    X_train : training points in scaled space (n_samples, n_dims)
    X1, X2 : 2D meshgrid arrays for the two visible dimensions
    mu_grid : predicted mean on the 2D grid (same shape as X1)
    sigma_grid : predicted std on the 2D grid
    dim1_idx, dim2_idx : int indices of the two dimensions being plotted
    X_excluded : optional excluded training points (n_samples, n_dims)
    title_prefix : string prepended to the suptitle
    fixed_info : string describing fixed dimension values, shown in suptitle
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    suptitle = f'{title_prefix}'
    if fixed_info:
        suptitle += f' | {fixed_info}'
    fig.suptitle(suptitle, fontsize=14)

    d1_label = f'$x_{{{dim1_idx+1}}}$'
    d2_label = f'$x_{{{dim2_idx+1}}}$'

    for ax in axes:
        ax.scatter(X_train[:, dim1_idx], X_train[:, dim2_idx],
                   c='red', marker='x', s=60, zorder=5, label='Training')
        if X_excluded is not None and len(X_excluded) > 0:
            ax.scatter(X_excluded[:, dim1_idx], X_excluded[:, dim2_idx],
                       c='black', marker='o', s=30, zorder=5, label='Excluded')

    cf0 = axes[0].contourf(X1, X2, mu_grid, levels=30, cmap='viridis')
    axes[0].set_title('Mean Prediction')
    axes[0].set_xlabel(d1_label)
    axes[0].set_ylabel(d2_label)
    fig.colorbar(cf0, ax=axes[0])
    axes[0].legend(fontsize=8)

    cf1 = axes[1].contourf(X1, X2, sigma_grid, levels=30, cmap='inferno')
    axes[1].set_title('Uncertainty (Std. Dev.)')
    axes[1].set_xlabel(d1_label)
    axes[1].set_ylabel(d2_label)
    fig.colorbar(cf1, ax=axes[1])
    axes[1].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()


def plot_nd_svm_analysis_slice(X1, X2, svm_proba, mu_svr_log, mu_svr_qt,
                                X_train, X_train_pos, X_train_neg, svm_labels,
                                dim1_idx, dim2_idx, fixed_info=''):
    """
    3-panel SVM analysis for an N-D problem, showing a 2D slice.

    Parameters:
    -----------
    X1, X2 : 2D meshgrid arrays for the two visible dimensions
    svm_proba : SVM classifier P(promising) on the 2D grid
    mu_svr_log : SVR log-space predictions on the 2D grid
    mu_svr_qt : SVR QT predictions on the 2D grid
    X_train : all training points in scaled space
    X_train_pos : positive training points
    X_train_neg : negative training points
    svm_labels : binary labels (1=promising, 0=not)
    dim1_idx, dim2_idx : int indices of the two visible dimensions
    fixed_info : string describing fixed dimension values
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    suptitle = 'SVM Analysis'
    if fixed_info:
        suptitle += f' | {fixed_info}'
    fig.suptitle(suptitle, fontsize=14)

    d1_label = f'$x_{{{dim1_idx+1}}}$ (scaled)'
    d2_label = f'$x_{{{dim2_idx+1}}}$ (scaled)'

    # Panel 1: SVM Classifier
    ax = axes[0]
    cf = ax.contourf(X1, X2, svm_proba, levels=20, cmap='RdYlGn')
    ax.scatter(X_train[svm_labels==1, dim1_idx], X_train[svm_labels==1, dim2_idx],
               c='green', s=150, edgecolors='k', label='Promising', zorder=5)
    ax.scatter(X_train[svm_labels==0, dim1_idx], X_train[svm_labels==0, dim2_idx],
               c='red', s=150, edgecolors='k', label='Not promising', zorder=5)
    ax.set_xlabel(d1_label)
    ax.set_ylabel(d2_label)
    ax.set_title('SVM Classifier: P(promising)')
    ax.legend(fontsize=8)
    fig.colorbar(cf, ax=ax)

    # Panel 2: SVR log-space
    ax = axes[1]
    cf = ax.contourf(X1, X2, mu_svr_log, levels=20, cmap='RdYlGn')
    ax.scatter(X_train_pos[:, dim1_idx], X_train_pos[:, dim2_idx],
               c='blue', s=150, edgecolors='k', zorder=5)
    ax.scatter(X_train_neg[:, dim1_idx], X_train_neg[:, dim2_idx],
               c='red', s=150, marker='^', edgecolors='k', zorder=5, label='Excluded')
    ax.set_xlabel(d1_label)
    ax.set_ylabel(d2_label)
    ax.set_title('SVR Surrogate: log10(y) [pos only]')
    ax.legend(fontsize=8)
    fig.colorbar(cf, ax=ax, label='log10(y)')

    # Panel 3: SVR QT
    ax = axes[2]
    cf = ax.contourf(X1, X2, mu_svr_qt, levels=20, cmap='RdYlGn')
    ax.scatter(X_train[:, dim1_idx], X_train[:, dim2_idx],
               c='blue', s=150, edgecolors='k', zorder=5)
    ax.set_xlabel(d1_label)
    ax.set_ylabel(d2_label)
    ax.set_title('SVR Surrogate: y_qt [all points]')
    fig.colorbar(cf, ax=ax, label='QuantileTransformer(y)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()


def plot_nd_acquisition_comparison_slice(X1, X2, surrogates, svm_proba,
                                          ensemble_ucb, X_train_pos, X_train_neg,
                                          dim1_idx, dim2_idx, fixed_info=''):
    """
    Acquisition function comparison for N-D, showing a 2D slice.

    Parameters:
    -----------
    X1, X2 : 2D meshgrid arrays
    surrogates : dict mapping name -> constrained UCB values (2D arrays)
    svm_proba : SVM P(promising) (2D array)
    ensemble_ucb : ensemble UCB (2D array)
    X_train_pos, X_train_neg : positive/negative training points
    dim1_idx, dim2_idx : int indices of visible dimensions
    fixed_info : string describing fixed dimension values
    """
    n_surrogates = len(surrogates)
    n_cols = max(n_surrogates, 2)
    fig, axes = plt.subplots(2, n_cols, figsize=(8 * n_cols, 12))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    suptitle = 'Acquisition Functions (SVM-constrained UCB)'
    if fixed_info:
        suptitle += f' | {fixed_info}'
    fig.suptitle(suptitle, fontsize=14)

    d1_label = f'$x_{{{dim1_idx+1}}}$'
    d2_label = f'$x_{{{dim2_idx+1}}}$'

    def _scatter_all(ax):
        ax.scatter(X_train_pos[:, dim1_idx], X_train_pos[:, dim2_idx],
                   c='r', marker='x', s=20, label='Positive')
        if len(X_train_neg) > 0:
            ax.scatter(X_train_neg[:, dim1_idx], X_train_neg[:, dim2_idx],
                       c='k', marker='o', s=10, label='Negative')

    for i, (name, ucb_vals) in enumerate(surrogates.items()):
        ax = axes[0, i]
        cf = ax.contourf(X1, X2, ucb_vals, levels=30, cmap='cividis')
        _scatter_all(ax)
        best_idx = np.unravel_index(np.argmax(ucb_vals), ucb_vals.shape)
        ax.plot(X1[best_idx], X2[best_idx], 'y*', markersize=15, label='Slice Max')
        ax.set_title(name)
        ax.set_xlabel(d1_label)
        ax.set_ylabel(d2_label)
        fig.colorbar(cf, ax=ax)
        ax.legend(fontsize=7)

    for i in range(n_surrogates, n_cols):
        axes[0, i].set_visible(False)

    # SVM
    ax = axes[1, 0]
    cf = ax.contourf(X1, X2, svm_proba, levels=20, cmap='Greens')
    ax.set_title('SVM P(Promising)')
    ax.set_xlabel(d1_label)
    ax.set_ylabel(d2_label)
    fig.colorbar(cf, ax=ax)

    # Ensemble
    ax = axes[1, 1]
    cf = ax.contourf(X1, X2, ensemble_ucb, levels=30, cmap='plasma')
    _scatter_all(ax)
    best_idx = np.unravel_index(np.argmax(ensemble_ucb), ensemble_ucb.shape)
    ax.plot(X1[best_idx], X2[best_idx], 'y*', markersize=15, label='Ensemble Max')
    ax.set_title('Ensemble UCB')
    ax.set_xlabel(d1_label)
    ax.set_ylabel(d2_label)
    fig.colorbar(cf, ax=ax)
    ax.legend(fontsize=7)

    for i in range(2, n_cols):
        axes[1, i].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.show()
