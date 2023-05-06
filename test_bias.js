importScripts("https://cdn.jsdelivr.net/pyodide/v0.22.1/full/pyodide.js");

function sendPatch(patch, buffers, msg_id) {
  self.postMessage({
    type: 'patch',
    patch: patch,
    buffers: buffers
  })
}

async function startApplication() {
  console.log("Loading pyodide!");
  self.postMessage({type: 'status', msg: 'Loading pyodide'})
  self.pyodide = await loadPyodide();
  self.pyodide.globals.set("sendPatch", sendPatch);
  console.log("Loaded!");
  await self.pyodide.loadPackage("micropip");
  const env_spec = ['https://cdn.holoviz.org/panel/0.14.3/dist/wheels/bokeh-2.4.3-py3-none-any.whl', 'https://cdn.holoviz.org/panel/0.14.3/dist/wheels/panel-0.14.3-py3-none-any.whl', 'pyodide-http==0.1.0', 'numpy', 'pandas', 'scikit-learn']
  for (const pkg of env_spec) {
    let pkg_name;
    if (pkg.endsWith('.whl')) {
      pkg_name = pkg.split('/').slice(-1)[0].split('-')[0]
    } else {
      pkg_name = pkg
    }
    self.postMessage({type: 'status', msg: `Installing ${pkg_name}`})
    try {
      await self.pyodide.runPythonAsync(`
        import micropip
        await micropip.install('${pkg}');
      `);
    } catch(e) {
      console.log(e)
      self.postMessage({
	type: 'status',
	msg: `Error while installing ${pkg_name}`
      });
    }
  }
  console.log("Packages loaded!");
  self.postMessage({type: 'status', msg: 'Executing code'})
  const code = `
  
import asyncio

from panel.io.pyodide import init_doc, write_doc

init_doc()

import numpy as np
import pandas as pd
import panel as pn

from bokeh.io import curdoc, output_notebook, push_notebook
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select
from bokeh.plotting import figure, show
from bokeh.models.widgets import Div

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor


def add_noise(y, noise_level=0.1):
    return y + np.random.normal(0, np.sqrt(noise_level), len(y))

def generate_simple_data(size=1000, random_state=None):
    np.random.seed(random_state)
    X = np.random.rand(size)
    y = 2 * X 
    return X.reshape(-1, 1), y

def generate_complex_data(size=1000, random_state=None):
    np.random.seed(random_state)
    X = np.random.rand(size)
    y = np.sin(2 * np.pi * X) 
    return X.reshape(-1, 1), y

def generate_very_complex_data(size=1000, random_state=None):
    np.random.seed(random_state)
    X = np.random.rand(size)
    y = np.sin(2 * np.pi * X) - 3*(X>0.5)
    return X.reshape(-1, 1), y

def fit_and_evaluate_model(model_name, data_name, X_train, y_train, X_test, y_test, boosting_regularization=0.1):
    model_mapping = {
        'Linear Regression': LinearRegression(),
        'Local Fit': KNeighborsRegressor(n_neighbors=5),
        'Polynomial': make_pipeline(PolynomialFeatures(degree=5), LinearRegression()),
        'Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=boosting_regularization)
    }
    model = model_mapping[model_name]

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    return y_train_pred, y_test_pred, mse_train, mse_test


# Define the update function
def update(attr, old, new):
    model_name = model_select.value
    data_name = data_select.value

    if data_name == 'Simple':
        X, y = generate_simple_data(random_state=42, size=int(size_select.value))
    elif data_name == 'Complex':
        X, y = generate_complex_data(random_state=42, size=int(size_select.value))
    else:
        X, y = generate_very_complex_data(random_state=42, size=int(size_select.value))

    y = add_noise(y, noise_level=float(noise_select.value))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Sort the train and test data based on X values
    train_indices = np.argsort(X_train, axis=0).flatten()
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    test_indices = np.argsort(X_test, axis=0).flatten()
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]


    y_train_pred, y_test_pred, mse_train, mse_test = fit_and_evaluate_model(model_name, data_name, X_train, y_train, X_test, y_test, boosting_regularization=float(regularization_select.value))

    source_train.data = dict(x=X_train.flatten(), y=y_train, y_pred=y_train_pred)
    source_test.data = dict(x=X_test.flatten(), y=y_test, y_pred=y_test_pred)

    mse_display.text = f"MSE (Train): {mse_train:.4f}<br>MSE (Test): {mse_test:.4f}"


# Create the data source
source_train = ColumnDataSource(data=dict(x=[], y=[], y_pred=[]))
source_test = ColumnDataSource(data=dict(x=[], y=[], y_pred=[]))

# Create the scatter plots
plot_train = figure(title='Training Data', x_axis_label='X', y_axis_label='y')
plot_train.scatter(x='x', y='y', source=source_train)
plot_train.line(x='x', y='y_pred', source=source_train, color='red')

plot_test = figure(title='Testing Data', x_axis_label='X', y_axis_label='y')
plot_test.scatter(x='x', y='y', source=source_test)
plot_test.line(x='x', y='y_pred', source=source_test, color='red')

# Create the widgets
model_select = Select(title='Model', value='Linear Regression', options=['Linear Regression', 'Local Fit', 'Polynomial', 'Boosting'])
data_select = Select(title='Dataset', value='Simple', options=['Simple', 'Complex', 'Very Complex'])
mse_display = Div(text="MSE (Train): N/A<br>MSE (Test): N/A")
regularization_select = Select(title='Boosting Regularization', value='0.1', options=['0.01', '0.1', '0.5', '1'])
size_select = Select(title='Sample size', value='100', options=['10', '100', '1000', '10000'])
noise_select = Select(title='Var(Noise)', value='0.1', options=['0.01', '0.1', '1', '10'])


# Add event listeners
model_select.on_change('value', update)
data_select.on_change('value', update)
regularization_select.on_change('value', update)
size_select.on_change('value', update)
noise_select.on_change('value', update)

# Set up the layout
layout = column(row(data_select, size_select, noise_select, model_select, regularization_select), row(plot_train, plot_test), mse_display)


# Initialize the plot
update(None, None, None)

bokeh_app = pn.pane.Bokeh(layout).servable()

# Add the layout to the app
#curdoc().add_root(layout)



await write_doc()
  `

  try {
    const [docs_json, render_items, root_ids] = await self.pyodide.runPythonAsync(code)
    self.postMessage({
      type: 'render',
      docs_json: docs_json,
      render_items: render_items,
      root_ids: root_ids
    })
  } catch(e) {
    const traceback = `${e}`
    const tblines = traceback.split('\n')
    self.postMessage({
      type: 'status',
      msg: tblines[tblines.length-2]
    });
    throw e
  }
}

self.onmessage = async (event) => {
  const msg = event.data
  if (msg.type === 'rendered') {
    self.pyodide.runPythonAsync(`
    from panel.io.state import state
    from panel.io.pyodide import _link_docs_worker

    _link_docs_worker(state.curdoc, sendPatch, setter='js')
    `)
  } else if (msg.type === 'patch') {
    self.pyodide.runPythonAsync(`
    import json

    state.curdoc.apply_json_patch(json.loads('${msg.patch}'), setter='js')
    `)
    self.postMessage({type: 'idle'})
  } else if (msg.type === 'location') {
    self.pyodide.runPythonAsync(`
    import json
    from panel.io.state import state
    from panel.util import edit_readonly
    if state.location:
        loc_data = json.loads("""${msg.location}""")
        with edit_readonly(state.location):
            state.location.param.update({
                k: v for k, v in loc_data.items() if k in state.location.param
            })
    `)
  }
}

startApplication()