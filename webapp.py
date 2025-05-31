import os
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
import subprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'results')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    images, csvs, params = None, None, None
    if request.method == 'POST':
        # Get form parameters
        agg_method = request.form.get('agg_method', 'krum')
        enable_attack = 'enable_attack' in request.form
        num_rounds = int(request.form.get('num_rounds', 10))
        use_robust_fl = 'use_robust_fl' in request.form
        num_adv = int(request.form.get('num_adv', 1))
        trim_ratio = float(request.form.get('trim_ratio', 0.1))
        attack_method = request.form.get('attack_method', 'random_noise')
        attack_epsilon = float(request.form.get('attack_epsilon', 10.0))
        use_migration_env = 'use_migration_env' in request.form
        num_edge_nodes = int(request.form.get('num_edge_nodes', 3))
        num_mobile_devices = int(request.form.get('num_mobile_devices', 5))
        num_services = int(request.form.get('num_services', 5))
        episodes_per_round = int(request.form.get('episodes_per_round', 100))

        # Clean previous results
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

        # Run the experiment (call main.py with arguments)
        cmd = [
            'python', 'src/main.py',
            '--agg_method', agg_method,
            '--enable_attack', str(enable_attack),
            '--attack_method', attack_method,
            '--attack_epsilon', str(attack_epsilon),
            '--use_robust_fl', str(use_robust_fl),
            '--num_adv', str(num_adv),
            '--trim_ratio', str(trim_ratio),
            '--use_migration_env', str(use_migration_env),
            '--num_edge_nodes', str(num_edge_nodes),
            '--num_mobile_devices', str(num_mobile_devices),
            '--num_services', str(num_services),
            '--episodes_per_round', str(episodes_per_round),
            '--num_rounds', str(num_rounds),
            '--output_dir', app.config['UPLOAD_FOLDER']
        ]
        print('Running command:', ' '.join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            flash('Experiment failed to run. Please check your code and parameters.', 'danger')
            return render_template('index.html', images=None, csvs=None, params=request.form)

        # Collect result files
        result_files = os.listdir(app.config['UPLOAD_FOLDER'])
        images = [f for f in result_files if f.endswith('.png')]
        csvs = [f for f in result_files if f.endswith('.csv')]
        params = request.form
        flash('Experiment completed successfully!', 'success')

    return render_template('index.html', images=images, csvs=csvs, params=params)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.secret_key = 'your_secret_key_here'
    app.run(debug=True) 