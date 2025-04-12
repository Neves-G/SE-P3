from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import os

# Initialize the app
app = Dash(__name__, assets_folder='assets', suppress_callback_exceptions=True)
app.title = "PV Panel Detection"
server = app.server

# Graph and table generation functions
def generate_table(dataframe, max_rows=10, style1={'width': '100%', 'tableLayout': 'auto'}, style2={'overflowX': 'auto', 'display': 'block', 'maxWidth': '100%'}):
    return html.Div([
        html.Div([
            html.Table([
                html.Thead(
                    html.Tr([html.Th(dataframe.index.name)] +
                            [html.Th(col) for col in dataframe.columns])
                ),
                html.Tbody([
                    html.Tr([html.Td(dataframe.index[i])] +
                            [html.Td(dataframe.iloc[i][col]) for col in dataframe.columns])
                    for i in range(min(len(dataframe), max_rows))
                ])
            ], style=style1)  # Allow flexible column widths
        ], style=style2)  # Horizontal scroll for the table only
    ])

def generate_graph(dataframe):
    return px.line(dataframe)

# Sample data for demonstration
testing_images = [f"/assets/Predictions_s/{path}" for path in os.listdir("dash-app/assets/Predictions_s")]  # Replace with actual paths
training_images = [f"/assets/Batch/{path}" for path in os.listdir("dash-app/assets/Batch")]  # Replace with actual paths

results_df = pd.read_csv("dash-app/run/Run_s/detect/train/results.csv")
results_df.rename(columns={"epoch": "Epoch"}, inplace=True)
results_df.set_index("Epoch", inplace=True)
results_df.drop(columns=["time"], inplace=True)

result_metrics = {
    "Full Metrics": "generate_graph(results_df)",
    "Confusion Matrix": "assets/Metrics/confusion_matrix.png",
    "Recall Curve": "assets/Metrics/R_curve.png",
    "Precision Curve": "assets/Metrics/P_curve.png",
    "PR Curve": "assets/Metrics/PR_curve.png",
    "F1 Curve": "assets/Metrics/F1_curve.png",
}

# Layout for each tab
home_layout = html.Div([
    html.H1("PV Panel Detection Project", style={"textAlign": "center", "marginTop": "20px"}),
    html.P("This project aims to detect photovoltaic panels using satellite imagery and a YOLOv11s model.", style={"textAlign": "center"}),
    html.P("Navigate through the tabs to explore the training data, results, and more!", style={"textAlign": "center"}),
    html.P(["Project made in the context of the ",
        html.B("Energy Servives"),
        " course at IST."
        ], style={"textAlign": "center"}),
    html.P([
            "This project was developed by ",
            html.A("Guilherme Neves", href="https://fenix.tecnico.ulisboa.pt/homepage/ist1102548", target="_blank", style={'color': '#FFB703', 'textDecoration': 'none'}),
            "."
        ], style={'textAlign': 'center'}),
    html.Div([
            html.Img(src='/assets/IST_A_RGB_POS.jpg', style={'height': '400px'})
        ], style={'textAlign': 'center'})
])

training_layout = html.Div([
    html.H1("Training Data", style={"textAlign": "left", "marginTop": "20px"}),

    # Left column: Textual information
    html.Div([
        html.P("The image selection was randomly generated using the Google Static Maps API, using satellite imagery from several Portuguese cities."),
        html.P([
            "After collection, the images were annotated via Roboflow, resulting in a 426-image dataset available ",
            html.A("here", href="https://app.roboflow.com/se-project-3/sat-panel-id/6", target="_blank", style={'color': '#FFB703', 'textDecoration': 'none'}),
            "."
        ]),
        html.P([
            html.B("Three"),
            " categories were annotated:"
            ]),
        html.Ol([
            html.Li("PV Panel"),
            html.Li("Skylight"),
            html.Li("Solar Thermal"),
        ], start=0),
        html.P("The dataset was split into 301 images for training, 50 for validation, and 75 for testing."),
        html.P("The training dataset was augmented in Roboflow with the following augmentations:"),
        html.Ul([
            html.Li([html.B("Shear:"),
                    " ±10° Horizontal, ±10° Vertical"
                ]),
            html.Li([html.B("Brightness:"),
                " Between -20% and +20%"     
                ]),
        ]),
        html.P("During training, the following augmentations were also applied:"),
        html.Ul([
            html.Li(html.B("Blur")),
            html.Li(html.B("Grayscale")),
            html.Li(html.B("CLAHE")),
        ]),
        html.P("The chosen model was YOLOv11s, trained over 100 epochs."),
    ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'left'}),

    # Right column: Image display
    html.Div([
        html.P("Below are some images from the training/validation dataset:"),
        html.Div(id="training-image-display"),
        html.Button("Previous", id="prev-training-image", n_clicks=0, style={"margin": "10px", 'textAlign': 'right'}),
        html.Button("Next", id="next-training-image", n_clicks=0, style={'textAlign': 'right'}),
        html.Div(id="training-image-info"),
    ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'center'}),
])

results_layout = html.Div([
    html.H1("Results", style={"textAlign": "left", "marginTop": "20px"}),

    # Left column: Metric selector and graph
    html.Div([
        html.P("Select a metric to view the corresponding results:"),
        dcc.RadioItems(
            id="metric-selector",
            options=[{"label": metric, "value": metric} for metric in result_metrics.keys()],
            value=list(result_metrics.keys())[0],
            style={"marginBottom": "10px"},
            inline=True,
        ),
        html.Div(id="metric-display", style={"marginBottom": "20px"}),
    ], style={'width': '60%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'left', 'justifyContent': 'center'}),

    # Right column: Image display
    html.Div([
        html.P("Below are some images from the test dataset:"),
        html.Div(id="result-image-display"),
        html.Button("Previous", id="prev-result-image", n_clicks=0, style={"margin": "10px", 'textAlign': 'right'}),
        html.Button("Next", id="next-result-image", n_clicks=0, style={'textAlign': 'right'}),
        html.Div(id="test-image-info"),
    ], style={'width': '40%', 'display': 'inline-block', 'verticalAlign': 'top', 'textAlign': 'center', 'justifyContent': 'center'}),
])

# Main app layout with tabs
app.layout = html.Div([
    dcc.Tabs(id="tabs", value="home", children=[
        dcc.Tab(label="Home", value="home"),
        dcc.Tab(label="Training", value="training"),
        dcc.Tab(label="Results", value="results"),
    ]),
    html.Div(id="tab-content"),
])

# Callbacks to update tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value")
)
def render_tab_content(tab_name):
    if tab_name == "home":
        return home_layout
    elif tab_name == "training":
        return training_layout
    elif tab_name == "results":
        return results_layout
    return html.Div("404 - Page not found")

# Callbacks for training page
@app.callback(
    [Output("training-image-display", "children"),
     Output("training-image-info", "children")],
    [Input("prev-training-image", "n_clicks"),
     Input("next-training-image", "n_clicks")],
)
def update_training_image_and_info(prev_clicks, next_clicks):
    index = int((next_clicks - prev_clicks) % len(training_images))
    image_name = os.path.basename(training_images[index])  # Extract the image name
    return (
        html.Img(src=training_images[index], style={"height": "400px", "width": "400px"}),
        html.P(f"Image: {image_name}", style={"textAlign": "center", "marginTop": "10px", 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
    )

# Callbacks for results page
@app.callback(
    Output("metric-display", "children"),
    Input("metric-selector", "value")
)
def update_metric_display(selected_metric):
    if selected_metric == "Full Metrics":
        return html.Div(
            dcc.Graph(figure=generate_graph(results_df), style={"height": "400px"})
            )
    else:
        return html.Div([
            html.Img(src=result_metrics[selected_metric], style={"height": "400px"}),
        ], style={'textAlign': 'center'})

@app.callback(
    [Output("result-image-display", "children"),
     Output("test-image-info", "children")],
    [Input("prev-result-image", "n_clicks"),
     Input("next-result-image", "n_clicks")],
)
def update_test_image_and_info(prev_clicks, next_clicks):
    index = int((next_clicks - prev_clicks) % len(testing_images))
    image_name = os.path.basename(testing_images[index])  # Extract the image name
    return (
        html.Img(src=testing_images[index], style={"height": "400px", "width": "400px"}),
        html.P(f"Image: {image_name}", style={"textAlign": "center", "marginTop": "10px", 'overflow': 'hidden', 'textOverflow': 'ellipsis', 'whiteSpace': 'nowrap'})
    )

if __name__ == "__main__":
    app.run()
