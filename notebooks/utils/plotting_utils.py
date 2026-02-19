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
            outlier_text += f'({X[idx][0]}, {X[idx][1]}): {y[idx]}\n'
        
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


def plot_2D_mean_uncertainty(X, model, X1_test, X2_test, mu, sigma):
    """
    Plot Bar chart to show outliers in the output
    
    Parameters:
    -----------
    X: input (2D)
    model: GaussianProcessRegressor
    X1_test: Prediction grid Dimension 1
    X2_test: Prediction grid Dimension 2
    mu
    sigma
    """

    mu_grid = mu.reshape(X1_test.shape)
    sigma_grid = sigma.reshape(X1_test.shape)

    # STEP 3: Plot
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 2D Mean
    ax_1 = fig.add_subplot(gs[0, 0])
    contour = ax_1.contourf(X1_test, X2_test, mu_grid, levels=20, cmap='RdYlGn')
    ax_1.scatter(X[:, 0], X[:, 1], c='blue', s=100, 
            edgecolors='black', linewidth=1.5, zorder=5)
    ax_1.set_xlabel('Input 1')
    ax_1.set_ylabel('Input 2')
    ax_1.set_title('GP Mean Prediction (Function Shape)')
    plt.colorbar(contour, ax=ax_1, label='Predicted Output')

    # 2D Uncertainty
    ax_2 = fig.add_subplot(gs[0, 1])
    contour = ax_2.contourf(X1_test, X2_test, sigma_grid, levels=20, cmap='YlOrRd')
    ax_2.scatter(X[:, 0], X[:, 1], c='blue', s=100, 
            edgecolors='black', linewidth=1.5, zorder=5)
    ax_2.set_xlabel('Input 1')
    ax_2.set_ylabel('Input 2')
    ax_2.set_title('GP Uncertainty (Where is GP Confident?)')
    plt.colorbar(contour, ax=ax_2, label='Uncertainty (Std Dev)')

    # 3D Mean
    ax_3 = fig.add_subplot(gs[1, 0], projection='3d')
    ax_3.plot_surface(X1_test, X2_test, mu_grid, cmap='RdYlGn')
    ax_3.set_xlabel('Input 1')
    ax_3.set_ylabel('Input 2')
    ax_3.set_zlabel('Predicted Output')
    ax_3.set_title('GP Mean Prediction (Function Shape) - 3D')
    ax_3.view_init(elev=25, azim=45)

    # 3D Uncertainty
    ax_4 = fig.add_subplot(gs[1, 1], projection='3d')
    ax_4.plot_surface(X1_test, X2_test, sigma_grid, cmap='YlOrRd')
    ax_4.set_xlabel('Input 1')
    ax_4.set_ylabel('Input 2')
    ax_4.set_zlabel('Uncertainty (Std)')
    ax_4.set_title('GP Uncertainty Surface (3D)')
    ax_4.view_init(elev=25, azim=45)

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

    # # 1. Prepare Data
    # n_samples, n_dims = X.shape
    # cols = [f'$x_{i+1}$' for i in range(n_dims)]
    # df = pd.DataFrame(X, columns=cols)
    # df['y'] = y
    
    # # Normalize data for plotting between 0 and 1 so axes are comparable
    # # But we keep the labels for the original scale
    # df_norm = (df - df.min()) / (df.max() - df.min())

    # fig = plt.figure(figsize=(18, 10))
    # gs = GridSpec(2, 2, height_ratios=[4, 2], figure=fig, hspace=0.4)
    # fig.suptitle(title, fontsize=16, fontweight='bold')


    # # Add Info Text
    # ax_info = fig.add_subplot(gs[0, 0])
    # ax_info.axis('off')
    # ax_info.text(0.01, 0.5, info_text, fontsize=10, family='monospace',
    #              bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.8))
    

    # # 3D graph
    # ax_1 = fig.add_subplot(gs[0, 1], projection='3d')
    
    # # Normalize y for marker size so they are visible but not exploding
    # # We use a min size of 20 and max of 300
    # y_scaled = ((y - y.min()) / (y.max() - y.min()) * 280) + 20

    # # Scatter plot
    # # c=X[:, 3] maps the 4th dimension to color
    # # s=y_scaled maps the output to size
    # img = ax_1.scatter(X[:, 0], X[:, 1], X[:, 2], 
    #                  c=X[:, 3], cmap='plasma', 
    #                  s=y_scaled, edgecolors='k', alpha=0.6)

    # ax_1.set_xlabel('$x_1$')
    # ax_1.set_ylabel('$x_2$')
    # ax_1.set_zlabel('$x_3$')
    
    # # Legends for the "hidden" dimensions
    # cbar = fig.colorbar(img, ax=ax_1, pad=0.1)
    # cbar.set_label('4th Dimension ($x_4$)', rotation=270, labelpad=15)
    
    # # Add a size legend proxy
    # for s in [y.min(), np.median(y), y.max()]:
    #     ax_1.scatter([], [], [], c='k', alpha=0.3, s=((s - y.min()) / (y.max() - y.min()) * 280) + 20,
    #                label=f'y = {s:.2f}')
    # ax_1.legend(title="Output Magnitude", loc='upper left')


    # # Parallel Coordinates Visualization
    # ax_2 = fig.add_subplot(gs[1, :])
    
    # # Create a colormap based on the output 'y'
    # cmap = plt.get_cmap('viridis')
    
    # # Plot each row as a line
    # for i in range(len(df_norm)):
    #     # Color line based on the normalized 'y' value
    #     color = cmap(df_norm.iloc[i]['y'])
    #     ax_2.plot(range(n_dims + 1), df_norm.iloc[i], color=color, alpha=0.5, linewidth=2)

    # # Aesthetics
    # ax_2.set_xticks(range(n_dims + 1))
    # ax_2.set_xticklabels(cols + ['Output ($y$)'], fontsize=12)
    # ax_2.set_yticks([]) # Vertical position is relative (0 to 1)
    # ax_2.set_title("Each line represents a single sample", pad=20)
    
    # # Add Colorbar
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=y.min(), vmax=y.max()))
    # cbar = fig.colorbar(sm, ax=ax_2, pad=0.05)
    # cbar.set_label('Target Value ($y$)', rotation=270, labelpad=15)

    # plt.show()

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