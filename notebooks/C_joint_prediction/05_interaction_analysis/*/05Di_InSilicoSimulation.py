import numpy as np
import pandas as pd
from sklearn.svm import SVR
from scipy.optimize import minimize
from scipy.stats import pearsonr
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
print2log = logger.info

def get_interaction_model(feature_1, 
                          feature_2, 
                          y,
                          svr_model, 
                          X_selected, 
                          feature_info):
    # make the feature_name  column in features_info the index
    feature_info = feature_info.set_index('gene_name')
    X_interaction = X_selected.copy()
    X_interaction = X_interaction.values
    feature_1_index = feature_info.loc[feature_1, 'feature_index']
    feature_2_index = feature_info.loc[feature_2, 'feature_index']
    interaction = X_interaction[:, feature_1_index]*X_interaction[:, feature_2_index]
    X_interaction = np.concatenate([X_interaction, interaction.reshape(-1, 1)], axis = 1)
    svr_model.fit(X_interaction, y)
    interaction_coef = svr_model.coef_[0, -1]
    # evaluate also training performance
    yhat = svr_model.predict(X_interaction)
    r = pearsonr(y.values.flatten(), yhat.flatten())[0]
    return interaction_coef, svr_model, r

def SynergyContourPlot(dataset, feature_pattern1, feature_pattern2, res_dir,name, save=True):
    # Interpolate onto a grid for smooth contours
    from scipy.interpolate import griddata
    xi = np.linspace(np.min(dataset.iloc[:, 0].values), np.max(dataset.iloc[:, 0].values), 100)
    yi = np.linspace(np.min(dataset.iloc[:, 1].values), np.max(dataset.iloc[:, 1].values), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((dataset.iloc[:, 0].values, dataset.iloc[:, 1].values), dataset.iloc[:, 2].values, (xi, yi), method='cubic')
    # Plot contour
    fig, ax = plt.subplots(figsize=(9, 9))
    contour = ax.contourf(xi, yi, zi, levels=100, cmap=sns.diverging_palette(240, 10, as_cmap=True),center=0)
    cbar = fig.colorbar(contour, ax=ax, label='Metastatic Potential')
    # Optional: add origin lines if origin in view
    #ax.axhline(0, color='black', linewidth=1)
    #ax.axvline(0, color='black', linewidth=1)
    # Axis labels and styling
    ax.set_xlabel(feature_pattern1, fontsize=16)
    ax.set_ylabel(feature_pattern2, fontsize=16)
    ax.tick_params(labelsize=12)
    ax.set_title("Contour of Metastatic Potential", fontsize=18)
    plt.tight_layout()
    # Save
    if save:
        plt.savefig(
            os.path.join(res_dir,name+"_"f"{feature_pattern1}_{feature_pattern2}_simulated_contour.png"),
            dpi=600,
            bbox_inches='tight'
        )
    return fig

def plot_grid_heatmap(grid_df, feat1_name, feat2_name, res_dir,name, save=True):
    """
    Pivot a grid DataFrame into a heatmap of the 'mean' column,
    sort axes ascending from negative to positive, set integer ticks,
    invert the y-axis, and optionally save the figure.
    Parameters
    ----------
    grid_df : pandas.DataFrame
        Must contain columns [feat1_name, feat2_name, 'mean'].
    feat1_name : str
        Column name to use for x-axis.
    feat2_name : str
        Column name to use for y-axis.
    res_dir : str
        Directory in which to save the figure (if save=True).
    name : str
        Name to use in figure filename.
    save : bool, default=True
        If True, saves the figure as PNG in res_dir.
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    """
    # 1) Pivot into a matrix form
    heatmap_data = grid_df.pivot(
        index=feat2_name,
        columns=feat1_name,
        values='mean'
    )
    # 2) Sort rows & columns ascending
    heatmap_data = heatmap_data.sort_index(axis=0, ascending=True)
    heatmap_data = heatmap_data.sort_index(axis=1, ascending=True)
    # 3) Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heatmap_data,
        cmap=sns.diverging_palette(240, 10, as_cmap=True),
        center=0,
        square=True,
        linewidths=0.5,
        linecolor='grey',
        cbar_kws={'label': 'Metastatic Potential'},
        ax=ax
    )
    # 4) Compute 10 evenly‐spaced integer ticks
    min_xi, max_xi = int(round(heatmap_data.columns.min())), int(round(heatmap_data.columns.max()))
    min_yi, max_yi = int(round(heatmap_data.index.min())),   int(round(heatmap_data.index.max()))
    x_labels = np.linspace(min_xi, max_xi, 10, dtype=int)
    y_labels = np.linspace(min_yi, max_yi, 10, dtype=int)
    x_pos    = np.linspace(0, heatmap_data.shape[1] - 1, 10, dtype=int)
    y_pos    = np.linspace(0, heatmap_data.shape[0] - 1, 10, dtype=int)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    # 5) Invert y-axis so low → bottom
    ax.invert_yaxis()
    # 6) Labels & layout
    ax.set_xlabel(feat1_name)
    ax.set_ylabel(feat2_name)
    plt.tight_layout()
    # 7) Save if requested
    if save:
        os.makedirs(res_dir, exist_ok=True)
        fname = name+"_"+f"{feat1_name}_{feat2_name}_simulated_heatmap.png"
        fig.savefig(os.path.join(res_dir, fname), dpi=600, bbox_inches='tight')
    return fig


def process_synergy_pair(
    feat1_name,
    feat2_name,
    X_selected,
    joint_features,
    model_template,
    Y,
    x_opt,
    n_steps,
    res_dir,
    save=True
):
    # 1) Train interaction model
    _, trained_svr_model,r = get_interaction_model(
        feat1_name, feat2_name, Y, model_template, X_selected, joint_features
    )

    # 2) Find column indices
    fi = joint_features.set_index('gene_name').loc[feat1_name, 'feature_index']
    fj = joint_features.set_index('gene_name').loc[feat2_name, 'feature_index']

    # 3) Build meshgrid over feature ranges
    f1_min = round(X_selected.values[:, fi].min()) - 1
    f1_max = round(X_selected.values[:, fi].max()) + 1
    f2_min = round(X_selected.values[:, fj].min()) - 1
    f2_max = round(X_selected.values[:, fj].max()) + 1
    f1_range = np.linspace(f1_min, f1_max, n_steps)
    f2_range = np.linspace(f2_min, f2_max, n_steps)
    G1, G2 = np.meshgrid(f1_range, f2_range)

    # 4) Tile baseline and overwrite two columns
    pts = G1.size
    X_grid = np.tile(x_opt, (pts, 1))
    X_grid[:, fi] = G1.ravel()
    X_grid[:, fj] = G2.ravel()
    interaction_col = (X_grid[:, fi] * X_grid[:, fj]).reshape(-1, 1)
    X_grid_int = np.concatenate([X_grid, interaction_col], axis=1)
    X_grid_base = np.concatenate([X_grid, np.zeros((pts, 1))], axis=1)

    # 5) Baseline predictions
    Z_base = trained_svr_model.predict(X_grid_base).reshape(G1.shape)
    grid_df_base = pd.DataFrame({
        feat1_name: G1.ravel(),
        feat2_name: G2.ravel(),
        'mean':     Z_base.ravel()
    })
    if save:
        os.makedirs(res_dir, exist_ok=True)
        grid_df_base.to_csv(
            os.path.join(res_dir, f"{feat1_name}_{feat2_name}_grid_base.csv"),
            index=False
        )
        print2log(f"saved baseline grid for {feat1_name}_{feat2_name}")

    # # 6) Plot baseline
    # SynergyContourPlot(grid_df_base, feat1_name, feat2_name, res_dir, name=f"{feat1_name}_{feat2_name}_base", save=save)
    # plot_grid_heatmap(grid_df_base, feat1_name, feat2_name, res_dir, name=f"{feat1_name}_{feat2_name}_base", save=save)

    # 7) Interaction predictions
    Z_int = trained_svr_model.predict(X_grid_int).reshape(G1.shape)
    grid_df_int = pd.DataFrame({
        feat1_name: G1.ravel(),
        feat2_name: G2.ravel(),
        'mean':     Z_int.ravel()
    })
    if save:
        grid_df_int.to_csv(
            os.path.join(res_dir, f"{feat1_name}_{feat2_name}_grid_int.csv"),
            index=False
        )
        print2log(f"saved interaction grid for {feat1_name}_{feat2_name}")

    # # 8) Plot interaction
    # SynergyContourPlot(grid_df_int, feat1_name, feat2_name, res_dir, name=f"{feat1_name}_{feat2_name}_int", save=save)
    # plot_grid_heatmap(grid_df_int, feat1_name, feat2_name, res_dir, name=f"{feat1_name}_{feat2_name}_int", save=save)

    # 9) Difference
    grid_df_diff = grid_df_int.copy()
    grid_df_diff['mean'] = grid_df_int['mean'] - grid_df_base['mean']
    if save:
        grid_df_diff.to_csv(
            os.path.join(res_dir, f"{feat1_name}_{feat2_name}_grid_diff.csv"),
            index=False
        )
        print2log(f"saved difference grid for {feat1_name}_{feat2_name}")

    # 10) Plot difference
    SynergyContourPlot(grid_df_diff, feat1_name, feat2_name, res_dir, name=f"{feat1_name}_{feat2_name}_diff", save=save)
    plot_grid_heatmap(grid_df_diff, feat1_name, feat2_name, res_dir, name=f"{feat1_name}_{feat2_name}_diff", save=save)
    return trained_svr_model,r


### Initialize the parsed arguments
parser = argparse.ArgumentParser(description='Run power analysis')
parser.add_argument('--X_all_path', action='store', type=str,help='path to the input training data',default='expr_joint.csv')
parser.add_argument('--Y_all_path', action='store', type=str,help='path to the output training data',default='metastatic_potential_joint.csv')
parser.add_argument('--interacting_features_path', action='store', type=str,help='path to the interacting features file',default='interactions_forNikos.csv')
parser.add_argument('--trained_features_path', action='store', type=str,help='path to the file containing selected features and trained model coeffiecients',default='joint_features.csv')
parser.add_argument('--model_parameters_path', action='store', type=str,help='path to the file containing model parameters',default='best_model_svr_linear_joint.pickle')
parser.add_argument('--interpolation_steps', action='store', type=int,help='Number of steps to use in interpolation for making a grid of values',default=100)
parser.add_argument('--predetermined_x_baseline', action='store', type=bool,help='A pre-determined gene expression baseline',default=True)
parser.add_argument('--predetermined_x_baseline_path', action='store', type=str,help='The path to a pre-determined gene expression baseline',default=None)
parser.add_argument('--center', action='store', type=bool,help='Center the data',default=False)
parser.add_argument('--res_dir', action='store', type=str,help='Results directory',default='synergy_analysis/')
args = parser.parse_args()
X_all_path = args.X_all_path
Y_all_path = args.Y_all_path
res_dir= args.res_dir
interacting_features_path = args.interacting_features_path
trained_features_path = args.trained_features_path
model_parameters_path = args.model_parameters_path
n_steps = args.interpolation_steps
predetermined_x_baseline = args.predetermined_x_baseline
predetermined_x_baseline_path = args.predetermined_x_baseline_path

if predetermined_x_baseline==True:
    if predetermined_x_baseline_path is None:
        raise ValueError(
            "If predetermined_x_baseline is True, "
            "predetermined_x_baseline_path must be specified"
        )

### Load the data
X = pd.read_csv(X_all_path,index_col=0)
Y = pd.read_csv(Y_all_path,index_col=0)
Y = Y.loc[X.index.values,["mean"]]
data = pd.concat([X, Y], axis=1, join='inner') 
joint_features = pd.read_csv(trained_features_path,index_col=0)
X_selected = X.loc[:,joint_features.feature_name.values]
if args.center==True:
    X_selected = X_selected.sub(X_selected.mean(axis=0), axis=1)
    print2log("Data were centered")

### Initialize model
baseline_model = SVR(kernel='linear', C=0.8708199642350806, epsilon=0.7444800190713263)
baseline_model.fit(X_selected, Y)

## Find which X gives zero metastatic potential
# Objective: minimize the predicted Y
def objective(x,mdl):
    x = np.array(x).reshape(1, -1)
    y_pred = mdl.predict(x)[0]
    return np.abs(y_pred)# y_pred**2  # minimizes y ≈ 0 , try also math.abs(y_pred)

input_dim = baseline_model.n_features_in_
# Initial guess: zero or random
x0 = np.zeros(input_dim)
# Optimize
if predetermined_x_baseline:
    x_opt = np.load(predetermined_x_baseline_path)
    y_val = baseline_model.predict(x_opt.reshape(1, -1))[0]
    print2log("X that minimizes Y ≈ 0:")
    print2log(x_opt)
    print2log("Predicted Y value:")
    print2log(y_val)
else:
    res = minimize(objective, x0,baseline_model, method='L-BFGS-B')
    # Result
    x_opt = res.x
    y_val = baseline_model.predict(x_opt.reshape(1, -1))[0]
    print2log("X that minimizes Y ≈ 0:")
    print2log(x_opt)
    print2log("Predicted Y value:")
    print2log(y_val)
    np.save('predetermined_x_baseline.npy',x_opt)
    print2log("Caclulated x_baseline is saved in "+res_dir+"predetermined_x_baseline.npy")

# ## plot
# plt.figure(figsize=(9, 9))
# plt.hist(X_selected.values.flatten(), bins=50)
# plt.show()

# plt.figure(figsize=(9, 9))
# plt.hist(x_opt.flatten(), bins=50)
# plt.show()

### Load interacting features
interacting_features = pd.read_csv(interacting_features_path)

# iterate through all interacting features
total_pairs = len(interacting_features)
train_r = []
for i, (_, row) in enumerate(
        tqdm(interacting_features.iterrows(),
             total=total_pairs,
             desc="Processing synergy pairs"),
        start=1):
    f1 = row["feature_1_gene_name"]
    f2 = row["feature_2_gene_name"]
    _,r = process_synergy_pair(
        feat1_name=f1,
        feat2_name=f2,
        X_selected=X_selected,
        joint_features=joint_features,
        model_template=SVR(kernel='linear', C=0.8708199642350806, epsilon=0.7444800190713263),
        Y=Y,
        x_opt=x_opt,
        n_steps=n_steps,
        res_dir=res_dir,
        save=True
    )
    train_r.append(r)
    print2log(f"Completed {i}/{total_pairs} synergy pairs")

## Plot in a histogram the training performance of the model
train_r = np.array(train_r)
fig = plt.subplots(figsize=(9, 9))
plt.hist(train_r, bins=10)
plt.xlabel("Training Performance", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.title("Histogram of Training Performance", fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(res_dir, "interaction_models_training_performance.png"), dpi=300)