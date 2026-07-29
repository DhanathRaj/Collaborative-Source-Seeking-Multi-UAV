clc;
clear;
close all;

%% ============================================================
% DISTRIBUTED GRADIENT-SEEKING FORMATION FLIGHT
% WITH ADAPTIVE NEURAL NETWORK ATTITUDE CONTROL
% + FOLLOWER GRADIENT ESTIMATION VIA PSEUDO-INVERSE
%
% Block Diagram Architecture:
%   Leader (Gradient-Based)
%     --> Formation/Position Controller  [e_p, e_v]
%     --> Gradient Estimator (Followers) [nabla_F_hat via pseudo-inverse]
%     --> Translational Control          [u_i]
%     --> Direction Mapping              [r_d = u/||u||]
%     --> Attitude Controller (ANN)      [tau_i]
%     --> Rotational Dynamics            [r_i, w_i]
%     --> UAV Dynamics                   [p_i, v_i]
%     --> (feedback) back to Formation Controller
%
% Follower Gradient Estimation:
%   Each follower i builds a local measurement matrix A and
%   field-difference vector delta_F using:
%     - Its own position x_i and field value F(x_i)
%     - The leader position x_L and field value F(x_L)
%     - Neighbouring followers (connectivity graph)
%   Then solves: A * nabla_F_hat = delta_F  via pseudo-inverse
%   The estimated gradient is blended into the translational command
%   to bias followers toward the source.
%
%% ============================================================

set(0,'DefaultAxesFontSize',12)
set(0,'DefaultLineLineWidth',1.8)
set(0,'DefaultFigureColor','w')

%% ============================================================
%% GLOBAL PARAMETERS
%% ============================================================

Nf = 4;          % number of follower UAVs
m  = 1.0;        % mass [kg]
g  = 9.81;       % gravity [m/s^2]
e3 = [0;0;1];    % unit vector

%% ---------------- INERTIA MATRIX ----------------

J = diag([0.1, 0.1, 0.1]);

%% ---------------- GAUSSIAN SOURCE FIELD ----------------

A_field = 1;
x0_field = [55; 60; 50];
sx = 14; sy = 18; sz = 16;

%% ---------------- SQUARE FORMATION OFFSETS ----------------

d_form = 3;
square_offsets = [ d_form,  d_form, 0;
                   d_form, -d_form, 0;
                  -d_form, -d_form, 0;
                  -d_form,  d_form, 0]';   % 3 x Nf

%% ---------------- TRANSLATIONAL CONTROLLER GAINS ----------------

kg_gain = 12.0;   % gradient ascent gain (leader)
kvL     = 3.5;   % leader velocity damping
kvG     = 3.0;   % global velocity damping

kx0     = 18.0;  % formation position gain
kv_form = 6.0;   % formation velocity gain

%% ---------------- FOLLOWER GRADIENT ESTIMATION GAIN ----------------
% Blends estimated gradient into follower translational command.
% Increase to make followers more aggressively seek the source.

kg_follow = 1.5;   % follower gradient-seeking gain

%% ---------------- ATTITUDE CONTROLLER GAINS (ANN-based) ----------------

kp    = 12;      % proportional attitude gain
kv1   = 2;       % sliding variable gain 1
kv2   = 0.3;     % sliding variable gain 2
k1    = 5;       % linear sliding term
k2    = 1;       % finite-time sliding term
alpha = 0.25;    % finite-time exponent

%% ---------------- FAST FINITE-TIME PARAMETERS ----------------

p_fft = 3;
q_fft = 1 - 1/p_fft;

%% ---------------- REGULARIZATION ----------------

eps1   = 0.05;
er_thr = 1e-3;

%% ---------------- TORQUE SATURATION ----------------

tau_max = 8;
fmax    = 30;     % thrust saturation

%% ---------------- ATTITUDE COUPLING GAINS ----------------

kr_att = 45.0;
kw_att = 18.0;

%% ---------------- NEURAL NETWORK PARAMETERS ----------------

n_input  = 6;
n_hidden = 80;
n_output = 3;

gammaW = 8;
gammaV = 6;

%% ---------------- NOISE PARAMETERS ----------------

sigma_v = 0.15;
sigma_w = 0.08;

%% ============================================================
%% STATE LAYOUT (per agent, including leader as agent 0)
%%
%% Leader  (index 0): [x(3), v(3), r(3), w(3)]            = 12 states
%% Follower i: [x(3), v(3), r(3), w(3),
%%              W_hat((n_hidden+1)*n_output),
%%              V_hat((n_input+1)*n_hidden)]
%%
%%   nn_size = (n_hidden+1)*n_output + (n_input+1)*n_hidden
%%   follower_dim = 12 + nn_size
%%
%% ============================================================

nn_size      = (n_hidden+1)*n_output + (n_input+1)*n_hidden;
follower_dim = 12 + nn_size;

%% ============================================================
%% INITIAL CONDITIONS
%% ============================================================

%% Leader
xL0 = [15;15;25];
vL0 = [0;0;0];
rL0 = e3;
wL0 = [0;0;0];

X0 = [xL0; vL0; rL0; wL0];   % leader block (12 states)

%% Followers
vel_var = [ 0.2,  0.3, 0.1;
            0.3,  0.2, 0.15;
            0.1,  0.2, 0.2;
            0.2,  0.3, 0.1]';

for i = 1:Nf
    xi0 = xL0 + square_offsets(:,i);
    vi0 = vel_var(:,i);
    ri0 = e3;
    wi0 = [0;0;0];

    W_hat0 = 0.02*randn(n_hidden+1, n_output);
    V_hat0 = 0.02*randn(n_input+1,  n_hidden);

    X0 = [X0;
          xi0; vi0; ri0; wi0;
          W_hat0(:);
          V_hat0(:)];
end

%% ============================================================
%% SIMULATION
%% ============================================================

Tend = 20;
opts = odeset('RelTol',1e-6,'AbsTol',1e-8,'MaxStep',0.02);

fprintf('Starting combined simulation...\n');
tic

%% Pack all parameters into struct for ODE function
P.Nf         = Nf;
P.m          = m;
P.g          = g;
P.e3         = e3;
P.J          = J;
P.A_field    = A_field;
P.x0_field   = x0_field;
P.sx = sx; P.sy = sy; P.sz = sz;
P.square_offsets = square_offsets;
P.kg_gain    = kg_gain;
P.kg_follow  = kg_follow;
P.kvL        = kvL;
P.kvG        = kvG;
P.kx0        = kx0;
P.kv_form    = kv_form;
P.kp         = kp;
P.kv1        = kv1;
P.kv2        = kv2;
P.k1         = k1;
P.k2         = k2;
P.alpha      = alpha;
P.q_fft      = q_fft;
P.eps1       = eps1;
P.er_thr     = er_thr;
P.tau_max    = tau_max;
P.fmax       = fmax;
P.kr_att     = kr_att;
P.kw_att     = kw_att;
P.n_input    = n_input;
P.n_hidden   = n_hidden;
P.n_output   = n_output;
P.nn_size    = nn_size;
P.follower_dim = follower_dim;
P.gammaW     = gammaW;
P.gammaV     = gammaV;
P.sigma_v    = sigma_v;
P.sigma_w    = sigma_w;

[t, X] = ode45(@(t,X) combined_dynamics(t, X, P), [0 Tend], X0, opts);

fprintf('Finished in %.2f seconds\n', toc);

%% ============================================================
%% DATA EXTRACTION
%% ============================================================

nT = length(t);

xL_hist = X(:, 1:3);
vL_hist = X(:, 4:6);
rL_hist = X(:, 7:9);
wL_hist = X(:, 10:12);

xF_hist            = zeros(nT, 3, Nf);
vF_hist            = zeros(nT, 3, Nf);
rF_hist            = zeros(nT, 3, Nf);
wF_hist            = zeros(nT, 3, Nf);
Delta_hat_hist     = zeros(nT, 3, Nf);
tau_norm_hist      = zeros(nT, Nf);
grad_est_hist      = zeros(nT, 3, Nf);   % estimated gradient per follower
grad_true_hist     = zeros(nT, 3);       % true gradient at leader
grad_est_err_hist  = zeros(nT, Nf);      % ||nabla_F_hat - nabla_F_true||

%% Helper: Gaussian field value and gradient
field_val  = @(x) A_field * exp(-0.5*( ...
    (x(1)-x0_field(1))^2/sx^2 + ...
    (x(2)-x0_field(2))^2/sy^2 + ...
    (x(3)-x0_field(3))^2/sz^2 ));
field_grad = @(x,F) F * [-(x(1)-x0_field(1))/sx^2; ...
                          -(x(2)-x0_field(2))/sy^2; ...
                          -(x(3)-x0_field(3))/sz^2];

for i = 1:Nf
    base = 12 + (i-1)*follower_dim;

    xF_hist(:,:,i) = X(:, base+(1:3));
    vF_hist(:,:,i) = X(:, base+(4:6));
    rF_hist(:,:,i) = X(:, base+(7:9));
    wF_hist(:,:,i) = X(:, base+(10:12));

    idxW_start = base + 12 + 1;
    idxW_end   = base + 12 + (n_hidden+1)*n_output;
    idxV_start = idxW_end + 1;
    idxV_end   = idxV_start + (n_input+1)*n_hidden - 1;

    for k = 1:nT
        ri  = rF_hist(k,:,i)';
        wi  = wF_hist(k,:,i)';
        rLk = rL_hist(k,:)';
        xi  = xF_hist(k,:,i)';
        xLk = xL_hist(k,:)';

        W_hat = reshape(X(k, idxW_start:idxW_end), [n_hidden+1, n_output]);
        V_hat = reshape(X(k, idxV_start:idxV_end), [n_input+1,  n_hidden]);

        %% Consensus error with leader
        e_r = cross(ri, rLk);

        %% Fast finite-time variable
        s   = e_r'*e_r + P.eps1;
        z_r = e_r / (s^P.q_fft);

        %% NN estimate
        x_nn   = [z_r; wi];
        x_bar  = [x_nn; 1];
        z_nn   = V_hat' * x_bar;
        sigma  = 1./(1+exp(-z_nn));
        sb     = [sigma; 1];
        Delta_hat = W_hat' * sb;

        Delta_hat_hist(k,:,i) = Delta_hat';

        %% Sliding variable
        phi       = wi + kv1*e_r + kv2*z_r;
        phi_norm  = norm(phi);
        fft_term  = phi / ((phi_norm + 0.05)^(1-alpha));

        %% Control torque
        tau = -kp*e_r - k1*J*phi - k2*J*fft_term - Delta_hat;
        tau_norm_hist(k,i) = norm(tau);

        %% ---- PSEUDO-INVERSE GRADIENT ESTIMATION (post-processing) ----
        %% Build measurement matrix using follower i and leader as two points.
        %% For post-processing use all Nf+1 available positions.

        F_xi = field_val(xi);
        F_xL = field_val(xLk);

        %% Collect positions and field values from all agents
        pos_all = [xLk, xF_hist(k,:,1)', xF_hist(k,:,2)', ...
                   xF_hist(k,:,3)', xF_hist(k,:,4)'];
        F_all   = zeros(5,1);
        F_all(1) = F_xL;
        for j = 1:4
            F_all(j+1) = field_val(xF_hist(k,:,j)');
        end

        %% Build A matrix: each row is (x_j - x_i)^T for j ~= i (agent i = follower idx i)
        ref_idx = i + 1;   % index in pos_all (1=leader, 2..5=followers)
        rows = setdiff(1:5, ref_idx);
        A_mat   = zeros(length(rows), 3);
        dF_vec  = zeros(length(rows), 1);
        for r = 1:length(rows)
            j = rows(r);
            dx = pos_all(:,j) - pos_all(:,ref_idx);
            A_mat(r,:) = dx';
            dF_vec(r)  = F_all(j) - F_all(ref_idx);
        end

        %% Pseudo-inverse solution: nabla_F_hat = A^+ * delta_F
        %% Regularised: (A'A + lambda*I)^{-1} A' delta_F
        lambda_reg = 1e-4;
        nabla_F_hat = (A_mat' * A_mat + lambda_reg * eye(3)) \ (A_mat' * dF_vec);

        grad_est_hist(k,:,i) = nabla_F_hat';

        %% True gradient at follower i position
        gF_true = field_grad(xi, F_xi);
        grad_true_hist(k,:)  = field_grad(xLk, F_xL)';   % at leader (reference)

        grad_est_err_hist(k,i) = norm(nabla_F_hat - gF_true);
    end
end

%% ============================================================
%% DERIVED METRICS
%% ============================================================

%% Formation errors (translational)
formation_err = zeros(nT, Nf);
for i = 1:Nf
    for k = 1:nT
        desired = xL_hist(k,:)' + square_offsets(:,i);
        formation_err(k,i) = norm(xF_hist(k,:,i)' - desired);
    end
end

%% Field value at leader
F_leader = zeros(nT,1);
for k = 1:nT
    xk = xL_hist(k,:)';
    F_leader(k) = field_val(xk);
end

dist_to_peak = sqrt(sum((xL_hist - x0_field').^2, 2));

%% Field values at each follower
F_follower = zeros(nT, Nf);
for i = 1:Nf
    for k = 1:nT
        F_follower(k,i) = field_val(xF_hist(k,:,i)');
    end
end

%% Attitude synchronization error (follower vs leader attitude)
att_sync_err = zeros(nT, Nf);
for i = 1:Nf
    for k = 1:nT
        ri  = rF_hist(k,:,i)';
        rLk = rL_hist(k,:)';
        att_sync_err(k,i) = norm(cross(ri, rLk));
    end
end

%% Velocity magnitudes
vL_mag = sqrt(sum(vL_hist.^2, 2));
vF_mag = zeros(nT, Nf);
for i = 1:Nf
    vF_mag(:,i) = sqrt(sum(vF_hist(:,:,i).^2, 2));
end

%% True gradient magnitude at leader (reference)
grad_true_mag = sqrt(sum(grad_true_hist.^2, 2));

%% Estimated gradient magnitude per follower
grad_est_mag = zeros(nT, Nf);
for i = 1:Nf
    grad_est_mag(:,i) = sqrt(sum(grad_est_hist(:,:,i).^2, 2));
end

%% ============================================================
%% PLOTTING — THESIS-QUALITY FIGURES
%% ============================================================

colors = lines(Nf);
follower_labels = {'Follower 1','Follower 2','Follower 3','Follower 4'};

%% ------ Figure 1: 3D Trajectory ------
fig1 = figure('Name','3D Trajectories','Position',[50 50 780 600]);
hold on; grid on; box on;
plot3(xL_hist(:,1), xL_hist(:,2), xL_hist(:,3), 'k-', 'LineWidth', 2.2, 'DisplayName','Leader');
for i = 1:Nf
    plot3(xF_hist(:,1,i), xF_hist(:,2,i), xF_hist(:,3,i), ...
        'Color', colors(i,:), 'LineWidth', 1.5, 'DisplayName', follower_labels{i});
end
scatter3(x0_field(1), x0_field(2), x0_field(3), 150, 'rp', 'filled', 'DisplayName','Source peak');
scatter3(xL_hist(1,1), xL_hist(1,2), xL_hist(1,3), 80, 'ko', 'filled', 'DisplayName','Start');
scatter3(xL_hist(end,1), xL_hist(end,2), xL_hist(end,3), 80, 'ks', 'filled', 'DisplayName','End');
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
title('3D Formation Trajectories');
legend('Location','best','FontSize',10);
view(35, 25);

%% ------ Figure 2: Formation Position Errors ------
fig2 = figure('Name','Formation Errors','Position',[50 50 780 420]);
hold on; grid on; box on;
for i = 1:Nf
    plot(t, formation_err(:,i), 'Color', colors(i,:), 'DisplayName', follower_labels{i});
end
xlabel('Time [s]'); ylabel('Position Error [m]');
title('Formation Position Error \|x_i - (x_L + d_i)\|');
legend('Location','best');
yline(0.1, 'k--', '0.1 m threshold', 'LabelHorizontalAlignment','right');

%% ------ Figure 3: Attitude Synchronization Errors ------
fig3 = figure('Name','Attitude Sync','Position',[50 50 780 420]);
hold on; grid on; box on;
for i = 1:Nf
    plot(t, att_sync_err(:,i), 'Color', colors(i,:), 'DisplayName', follower_labels{i});
end
xlabel('Time [s]'); ylabel('\|r_i \times r_L\|');
title('Attitude Synchronization Error (Follower vs Leader)');
legend('Location','best');

%% ------ Figure 4: Source Field Convergence ------
fig4 = figure('Name','Field Convergence','Position',[50 50 780 500]);

subplot(2,1,1);
hold on; grid on; box on;
plot(t, F_leader, 'k-', 'LineWidth', 2.2, 'DisplayName','Leader');
for i = 1:Nf
    plot(t, F_follower(:,i), 'Color', colors(i,:), 'DisplayName', follower_labels{i});
end
yline(A_field, 'r--', 'Peak F = 1', 'LabelHorizontalAlignment','right');
xlabel('Time [s]'); ylabel('F(x)');
title('Gaussian Field Value at Each Agent');
legend('Location','best','FontSize',9);

subplot(2,1,2);
plot(t, dist_to_peak, 'k-', 'LineWidth', 2.2);
grid on; box on;
xlabel('Time [s]'); ylabel('Distance to Peak [m]');
title('Leader Distance to Source Peak');

%% ------ Figure 5: ANN Disturbance Estimation ------
Delta_true_hist = zeros(nT, 3);
for k = 1:nT
    Delta_true_hist(k,:) = [0.5*sin(2*t(k)); 0.4*cos(1.5*t(k)); 0.3*sin(3*t(k))]';
end

fig5 = figure('Name','ANN Disturbance','Position',[50 50 900 560]);
comp_labels = {'x-axis','y-axis','z-axis'};
comp_colors = {'b','r','g'};
for c = 1:3
    subplot(3,1,c);
    hold on; grid on; box on;
    plot(t, Delta_true_hist(:,c), 'k--', 'LineWidth', 1.8, 'DisplayName','\Delta_{true}');
    for i = 1:Nf
        plot(t, Delta_hat_hist(:,c,i), 'Color', colors(i,:), ...
             'DisplayName', follower_labels{i});
    end
    ylabel(['\Delta_{' comp_labels{c} '} [N\cdotm]']);
    title(['ANN Disturbance Estimate — ' comp_labels{c}]);
    if c == 1; legend('Location','best','FontSize',9); end
    if c == 3; xlabel('Time [s]'); end
end
sgtitle('ANN Disturbance Estimation vs True Disturbance', 'FontWeight', 'bold');

%% ------ Figure 6: Control Torque Norms ------
fig6 = figure('Name','Torque Norms','Position',[50 50 780 420]);
hold on; grid on; box on;
for i = 1:Nf
    plot(t, tau_norm_hist(:,i), 'Color', colors(i,:), 'DisplayName', follower_labels{i});
end
yline(tau_max, 'k--', '\tau_{max}', 'LabelHorizontalAlignment','right');
xlabel('Time [s]'); ylabel('\|\tau_i\| [N\cdotm]');
title('Control Torque Norm per Follower');
legend('Location','best');

%% ------ Figure 7: Pseudo-Inverse Gradient Estimation ------
fig7 = figure('Name','Gradient Estimation','Position',[50 50 900 600]);

subplot(2,2,1);
hold on; grid on; box on;
plot(t, grad_true_mag, 'k-', 'LineWidth', 2.2, 'DisplayName','True |\nablaF| at leader');
for i = 1:Nf
    plot(t, grad_est_mag(:,i), 'Color', colors(i,:), 'DisplayName', follower_labels{i});
end
xlabel('Time [s]'); ylabel('|\nablaF| [(field/m)]');
title('Gradient Magnitude: Estimated vs True');
legend('Location','best','FontSize',9);

subplot(2,2,2);
hold on; grid on; box on;
for i = 1:Nf
    semilogy(t, max(grad_est_err_hist(:,i), 1e-8), 'Color', colors(i,:), ...
             'DisplayName', follower_labels{i});
end
xlabel('Time [s]'); ylabel('|\nabla\hat{F} - \nablaF| (log)');
title('Gradient Estimation Error (semi-log)');
legend('Location','best','FontSize',9);
grid on; box on;

%% Per-axis gradient components for follower 1 vs truth
grad_comps = {'x','y','z'};
for c = 1:3
    subplot(2,3,3+c);
    hold on; grid on; box on;
    true_comp = zeros(nT,1);
    for k = 1:nT
        xLk = xL_hist(k,:)';
        F_xL = field_val(xLk);
        g_true = field_grad(xLk, F_xL);
        true_comp(k) = g_true(c);
    end
    plot(t, true_comp, 'k--', 'LineWidth', 2, 'DisplayName','True');
    for i = 1:Nf
        plot(t, grad_est_hist(:,c,i), 'Color', colors(i,:), ...
             'DisplayName', follower_labels{i});
    end
    xlabel('Time [s]');
    ylabel(['\partialF/\partial' grad_comps{c}]);
    title(['Gradient component — ' grad_comps{c}]);
    if c == 1; legend('Location','best','FontSize',8); end
end
sgtitle('Pseudo-Inverse Gradient Estimation Results', 'FontWeight', 'bold');

%% ------ Figure 8: Velocity Magnitudes ------
fig8 = figure('Name','Velocities','Position',[50 50 780 420]);
hold on; grid on; box on;
plot(t, vL_mag, 'k-', 'LineWidth', 2.2, 'DisplayName','Leader');
for i = 1:Nf
    plot(t, vF_mag(:,i), 'Color', colors(i,:), 'DisplayName', follower_labels{i});
end
xlabel('Time [s]'); ylabel('Speed [m/s]');
title('Agent Speed Profiles');
legend('Location','best');

%% ------ Figure 9: Summary Dashboard ------
fig9 = figure('Name','Summary Dashboard','Position',[50 50 1100 700]);

subplot(2,3,1);
hold on; grid on; box on;
for i = 1:Nf
    plot(t, formation_err(:,i), 'Color', colors(i,:));
end
xlabel('t [s]'); ylabel('[m]'); title('Formation error');
yline(0.1,'k--');

subplot(2,3,2);
hold on; grid on; box on;
for i = 1:Nf
    plot(t, att_sync_err(:,i), 'Color', colors(i,:));
end
xlabel('t [s]'); ylabel('[\cdot]'); title('Attitude sync error');

subplot(2,3,3);
hold on; grid on; box on;
plot(t, F_leader, 'k-', 'LineWidth', 2);
for i = 1:Nf
    plot(t, F_follower(:,i), 'Color', colors(i,:));
end
yline(1,'r--'); xlabel('t [s]'); ylabel('F(x)'); title('Field value');

subplot(2,3,4);
hold on; grid on; box on;
for i = 1:Nf
    semilogy(t, max(grad_est_err_hist(:,i),1e-8), 'Color', colors(i,:));
end
xlabel('t [s]'); ylabel('error (log)'); title('Gradient est. error');

subplot(2,3,5);
hold on; grid on; box on;
for i = 1:Nf
    plot(t, tau_norm_hist(:,i), 'Color', colors(i,:));
end
yline(tau_max,'k--');
xlabel('t [s]'); ylabel('[N\cdotm]'); title('Torque norm');

subplot(2,3,6);
hold on; grid on; box on;
est_err_total = sum(grad_est_err_hist, 2) / Nf;
true_g_mag = grad_true_mag;
yyaxis left
  plot(t, est_err_total, 'b-');
  ylabel('Mean est. error');
yyaxis right
  plot(t, true_g_mag, 'r--');
  ylabel('|\nablaF|_{true}');
xlabel('t [s]'); title('Gradient est. quality');
legend({'Mean error','|\nablaF|'},'Location','best','FontSize',8);

sgtitle('Simulation Summary Dashboard', 'FontWeight', 'bold', 'FontSize', 14);

%% ============================================================
%% COMBINED DYNAMICS FUNCTION
%% ============================================================

function dX = combined_dynamics(t, X, P)

%% ----- Persistent noise (sample-and-hold at 20 Hz) -----
persistent noise_v noise_w last_t
if isempty(last_t) || (t - last_t) >= 0.05
    noise_v = P.sigma_v * [randn; randn; 0];
    noise_w = P.sigma_w * randn(3,1);
    last_t  = t;
end

%% ----- Allocate output -----
dX = zeros(size(X));

%% ============================================================
%% BLOCK 1: LEADER (GRADIENT-BASED)
%% ============================================================

xL = X(1:3);
vL = X(4:6);
rL = X(7:9);
wL = X(10:12);

rL = rL / max(norm(rL), 1e-6);

F_L = P.A_field * exp(-0.5*( ...
    (xL(1)-P.x0_field(1))^2/P.sx^2 + ...
    (xL(2)-P.x0_field(2))^2/P.sy^2 + ...
    (xL(3)-P.x0_field(3))^2/P.sz^2 ));

grad_L = F_L * [-(xL(1)-P.x0_field(1))/P.sx^2;
                -(xL(2)-P.x0_field(2))/P.sy^2;
                -(xL(3)-P.x0_field(3))/P.sz^2];

ghat  = grad_L / (norm(grad_L) + 1e-6);
scale = min(1, norm(xL - P.x0_field)/20 + 0.3);

uL = P.m*(P.kg_gain*scale*ghat - P.kvL*vL) + P.m*P.g*P.e3;
fL = min(norm(uL), P.fmax);
rLd = uL / (norm(uL) + 1e-6);

dxL = vL;
dvL = -P.g*P.e3 + (fL/P.m)*rL - P.kvG*vL + noise_v;
drL = cross(rL, wL);
dwL = -P.kr_att*cross(rL, rLd) - P.kw_att*wL + noise_w;

dX(1:3)   = dxL;
dX(4:6)   = dvL;
dX(7:9)   = drL;
dX(10:12) = dwL;

aL = dvL;

%% ============================================================
%% BLOCK 2-6: FOLLOWERS — Gradient Estimation + Formation +
%%             ANN Attitude Controller
%% ============================================================

%% Pre-compute field values at leader (used by all followers)
%% for pseudo-inverse gradient estimation inside ODE

F_field = @(x) P.A_field * exp(-0.5*( ...
    (x(1)-P.x0_field(1))^2/P.sx^2 + ...
    (x(2)-P.x0_field(2))^2/P.sy^2 + ...
    (x(3)-P.x0_field(3))^2/P.sz^2 ));

F_xL = F_field(xL);

%% Collect all follower positions and field values for gradient estimation
pos_all = zeros(3, P.Nf+1);
F_all   = zeros(P.Nf+1, 1);
pos_all(:,1) = xL;
F_all(1)     = F_xL;
for i = 1:P.Nf
    base_i = 12 + (i-1)*P.follower_dim;
    xi_i   = X(base_i+(1:3));
    pos_all(:,i+1) = xi_i;
    F_all(i+1)     = F_field(xi_i);
end

for i = 1:P.Nf

    base = 12 + (i-1)*P.follower_dim;

    xi = X(base+(1:3));
    vi = X(base+(4:6));
    ri = X(base+(7:9));
    wi = X(base+(10:12));

    ri = ri / max(norm(ri), 1e-6);

    idxW_s = base + 12 + 1;
    idxW_e = base + 12 + (P.n_hidden+1)*P.n_output;
    idxV_s = idxW_e + 1;
    idxV_e = idxV_s + (P.n_input+1)*P.n_hidden - 1;

    W_hat = reshape(X(idxW_s:idxW_e), [P.n_hidden+1, P.n_output]);
    V_hat = reshape(X(idxV_s:idxV_e), [P.n_input+1,  P.n_hidden]);

    %% ---- PSEUDO-INVERSE GRADIENT ESTIMATION ----
    %% Agent i uses measurements from all other agents to estimate ∇F at xi.
    %% Measurement model (linear approximation of Gaussian field):
    %%   F(x_j) - F(x_i) ≈ ∇F(x_i)^T * (x_j - x_i)
    %% Stacked as:  A_mat * nabla_F_hat = delta_F
    %% Solution:    nabla_F_hat = (A'A + λI)^{-1} A' delta_F   (Tikhonov reg.)

    ref_idx = i + 1;    % follower i is column (i+1) in pos_all
    all_idx = 1:(P.Nf+1);
    nbr_idx = all_idx(all_idx ~= ref_idx);  % all other agents

    n_nbr   = length(nbr_idx);
    A_mat   = zeros(n_nbr, 3);
    dF_vec  = zeros(n_nbr, 1);

    for r = 1:n_nbr
        j = nbr_idx(r);
        dx = pos_all(:,j) - pos_all(:,ref_idx);
        A_mat(r,:) = dx';
        dF_vec(r)  = F_all(j) - F_all(ref_idx);
    end

    lambda_reg   = 1e-4;   % Tikhonov regularisation
    nabla_F_hat  = (A_mat'*A_mat + lambda_reg*eye(3)) \ (A_mat'*dF_vec);

    %% Normalize estimated gradient direction
    grad_hat_norm = norm(nabla_F_hat);
    if grad_hat_norm > 1e-6
        ghat_i = nabla_F_hat / grad_hat_norm;
    else
        ghat_i = zeros(3,1);
    end

    %% Adaptive scale: reduce gradient influence near the source
    scale_i = min(1, norm(xi - P.x0_field)/20 + 0.2);

    %% ---- FORMATION / POSITION CONTROLLER ----
    e_p = xi - (xL + P.square_offsets(:,i));
    e_v = vi - vL;

    %% ---- TRANSLATIONAL CONTROL (blended with gradient estimate) ----
    %% u_i = m*a_L - kx0*e_p - kv*e_v + m*g*e3 + m*kg_follow*ghat_i
    ui = P.m*aL ...
       - P.kx0*e_p ...
       - P.kv_form*e_v ...
       + P.m*P.g*P.e3 ...
       + P.m*P.kg_follow*scale_i*ghat_i;   % <-- gradient bias term

    fi = min(norm(ui), P.fmax);

    %% ---- DIRECTION MAPPING ----
    rid = ui / (norm(ui) + 1e-6);

    %% ---- ATTITUDE CONSENSUS ERROR ----
    e_r = cross(ri, rL);
    e_r_dot = cross(cross(ri,wi), rL) + cross(ri, cross(rL, wL));

    %% ---- FAST FINITE-TIME VARIABLE ----
    s   = e_r'*e_r + P.eps1;
    z_r = e_r / (s^P.q_fft);

    if norm(e_r) > P.er_thr
        z_r_dot = e_r_dot/(s^P.q_fft) ...
                - 2*P.q_fft*e_r*(e_r'*e_r_dot)/(s^(P.q_fft+1));
    else
        z_r_dot = zeros(3,1);
    end

    %% ---- SLIDING VARIABLE ----
    phi = wi + P.kv1*e_r + P.kv2*z_r;

    %% ---- FAST FINITE-TIME TERM ----
    phi_norm = norm(phi);
    fft_term = phi / ((phi_norm + 0.05)^(1-P.alpha));

    %% ---- NEURAL NETWORK (DISTURBANCE ESTIMATOR) ----
    x_nn  = [z_r; wi];
    x_bar = [x_nn; 1];
    z_nn  = V_hat' * x_bar;
    sigma = 1./(1+exp(-z_nn));
    dsig  = sigma.*(1-sigma);
    sb    = [sigma; 1];

    Delta_hat = W_hat' * sb;

    %% ---- ATTITUDE CONTROLLER (ANN-enhanced) ----
    Delta_true = [ 0.5*sin(2*t) ;
                   0.4*cos(1.5*t);
                   0.3*sin(3*t) ];

    att_geo_err = cross(ri, rid);

    tau = cross(wi, P.J*wi) ...
         - P.J*(P.kv1*e_r_dot + P.kv2*z_r_dot) ...
         - P.kp*(e_r + att_geo_err) ...
         - P.k1*P.J*phi ...
         - P.k2*P.J*fft_term ...
         - Delta_hat;

    tau = max(min(tau, P.tau_max), -P.tau_max);

    %% ---- ROTATIONAL DYNAMICS ----
    w_dot = P.J \ (-cross(wi, P.J*wi) + tau + Delta_true);

    %% ---- ATTITUDE KINEMATICS ----
    r_dot = cross(ri, wi);
    r_dot = r_dot - ri*(ri.'*r_dot);

    %% ---- TRANSLATIONAL DYNAMICS ----
    dxi = vi;
    dvi = -P.g*P.e3 + (fi/P.m)*ri - P.kvG*vi;

    %% ---- NN WEIGHT UPDATE LAWS ----
    W_dot = P.gammaW * (sb * phi');

    W_no_bias = W_hat(1:P.n_hidden, :);
    temp = dsig .* (W_no_bias * phi);
    V_dot = P.gammaV * (x_bar * temp');

    %% ---- STORE DERIVATIVES ----
    dX(base+(1:3))   = dxi;
    dX(base+(4:6))   = dvi;
    dX(base+(7:9))   = r_dot;
    dX(base+(10:12)) = w_dot;

    dX(idxW_s:idxW_e) = W_dot(:);
    dX(idxV_s:idxV_e) = V_dot(:);

end

end