from flask import Blueprint, render_template
from .analysis import get_dataframe_summary, generate_plots, generate_3d_plot

main = Blueprint('main', __name__)

@main.route('/')
def index():
    summary_df = get_dataframe_summary()
    generate_plots()
    plotly_3d_json = generate_3d_plot()

    images = [
        "static/plots/sales_by_country.png",
        "static/plots/correlation_matrix.png",
        "static/plots/clusters_2d.png",
        "static/plots/histograms_cluster.png"
    ]
    return render_template("index.html",
                       table=summary_df.to_html(classes="table table-striped table-hover", index=False),
                       images=images,
                       plotly_3d=plotly_3d_json)

