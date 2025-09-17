clear all; close all; clc; 
dom_mark_size = 200;
non_mark_size = 200;
lw=1;
marker_lw=0.3;

%% Values obtained from extract_J1_J2_from_protocol.py

J_unbal_standard_A = [10.724323526940651, 46.0985652252444];
J_unbal_standard_D = [3.5405345951828107, 46.0985652252444];
J_N10_unbal_1_high_A_A = [6.475372555928081, 32.852856316756096];
J_N10_unbal_1_high_A_D = [-5.444357873394012, 32.852856316756096];
J_N10_unbal_1_low_A_A = [6.492359787432543, 36.15689575211856];
J_N10_unbal_1_low_A_D = [-5.273063270374891, 36.15689575211856];
J_N10_unbal_1_high_D_A = [6.618319437194358, 65.16278886278239];
J_N10_unbal_1_high_D_D = [-6.632568804912559, 65.16278886278239];
J_N10_unbal_1_low_D_A = [6.649638287689042, 49.37978496280422];
J_N10_unbal_1_low_D_D = [-6.665868175394013, 49.37978496280422];

cmap_balance = [0, 1, 1; 0, 0.5, 0.5; 0, 0.2, 0.2];
cmap_N = colormap(cool(6));

%% Load data

cell_name_list_A = ["N5_unbalanced_high", "N6_unbalanced_high", "N7_unbalanced_high",...
    "N8_unbalanced_high", "N9_unbalanced_high", "N10_unbalanced_high",...
    "N5_unbalanced_low", "N6_unbalanced_low", "N7_unbalanced_low", "N8_unbalanced_low", "N9_unbalanced_low", "N10_unbalanced_low"];

N_list_A = [5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10];

C_A = cell(length(cell_name_list_A), 5);

cell_name_list_D = ["N5_unbalanced_high", "N6_unbalanced_high", "N7_unbalanced_high",...
    "N8_unbalanced_high", "N9_unbalanced_high", "N10_unbalanced_high",...
    "N5_unbalanced_low", "N6_unbalanced_low", "N7_unbalanced_low", "N8_unbalanced_low", "N9_unbalanced_low", "N10_unbalanced_low"];

N_list_D = [5, 6, 7, 8, 9, 10, 5, 6, 7, 8, 9, 10];

C_D = cell(length(cell_name_list_D), 5);


for j = 1:length(N_list_A)
    C_A{j,1} = cell_name_list_A{j};
    FID1 = fopen("Pareto_results/pareto_A_" + C_A(j,1) + "_multi/pareto_CIET_A_200mV_50mV_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0.txt");
    out1 = textscan(FID1, '%f %f');
    fclose(FID1);
    pareto_output = cell2mat(out1);
    
    FID2 = fopen("Pareto_results/pareto_A_" + C_A(j,1) + "_multi/optimized_output_CIET_A_200mV_50mV_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0.txt");
    out2 = textscan(FID2, repmat('%f ', 1, 2*N_list_A(j)));
    fclose(FID2);
    optimized_output = cell2mat(out2);

    [~, p] = sort(pareto_output(:,2));
    C_A{j,2} = pareto_output(p,:);
    C_A{j,3} = optimized_output(p, 1:N_list_A(j));
    C_A{j,4} = optimized_output(p, (N_list_A(j)+1):(2*N_list_A(j)));
    
end

for j = 1:length(N_list_D)
    C_D{j,1} = cell_name_list_D{j};
    FID1 = fopen("Pareto_results/pareto_D_" + C_D(j,1) + "_multi/pareto_CIET_D_200mV_50mV_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0.txt");
    out1 = textscan(FID1, '%f %f');
    fclose(FID1);
    pareto_output = cell2mat(out1);
    
    FID2 = fopen("Pareto_results/pareto_D_" + C_D(j,1) + "_multi/optimized_output_CIET_D_200mV_50mV_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0.txt");
    out2 = textscan(FID2, repmat('%f ', 1, 2*N_list_D(j)));
    fclose(FID2);
    optimized_output = cell2mat(out2);

    [~, p] = sort(pareto_output(:,2));
    C_D{j,2} = pareto_output(p,:);
    C_D{j,3} = optimized_output(p, 1:N_list_D(j));
    C_D{j,4} = optimized_output(p, (N_list_D(j)+1):(2*N_list_D(j)));
    
end


%% Plot pareto optimal frontier
fig = figure(); clf(); 
scatter(C_A{1,2}(:,2), C_A{1,2}(:,1), dom_mark_size, cmap_N(1,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=5/high')
hold on
plot(C_A{1,2}(:,2), C_A{1,2}(:,1), 'Color', cmap_N(1,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_A{2,2}(:,2), C_A{2,2}(:,1), dom_mark_size, cmap_N(2,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=6/high')
plot(C_A{2,2}(:,2), C_A{2,2}(:,1), 'Color', cmap_N(2,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_A{3,2}(:,2), C_A{3,2}(:,1), dom_mark_size, cmap_N(3,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=7/high')
plot(C_A{3,2}(:,2), C_A{3,2}(:,1), 'Color', cmap_N(3,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_A{4,2}(:,2), C_A{4,2}(:,1), dom_mark_size, cmap_N(4,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=8/high')
plot(C_A{4,2}(:,2), C_A{4,2}(:,1), 'Color', cmap_N(4,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_A{5,2}(:,2), C_A{5,2}(:,1), dom_mark_size, cmap_N(5,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=9/high')
plot(C_A{5,2}(:,2), C_A{5,2}(:,1), 'Color', cmap_N(5,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_A{6,2}(:,2), C_A{6,2}(:,1), dom_mark_size, cmap_N(6,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=10/high')
plot(C_A{6,2}(:,2), C_A{6,2}(:,1), 'Color', cmap_N(6,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_A{7,2}(:,2), C_A{7,2}(:,1), dom_mark_size, cmap_N(1,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=5/low')
plot(C_A{7,2}(:,2), C_A{7,2}(:,1), 'Color', cmap_N(1,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_A{8,2}(:,2), C_A{8,2}(:,1), dom_mark_size, cmap_N(2,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=6/low')
plot(C_A{8,2}(:,2), C_A{8,2}(:,1), 'Color', cmap_N(2,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_A{9,2}(:,2), C_A{9,2}(:,1), dom_mark_size, cmap_N(3,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=7/low')
plot(C_A{9,2}(:,2), C_A{9,2}(:,1), 'Color', cmap_N(3,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_A{10,2}(:,2), C_A{10,2}(:,1), dom_mark_size, cmap_N(4,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=8/low')
plot(C_A{10,2}(:,2), C_A{10,2}(:,1), 'Color', cmap_N(4,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_A{11,2}(:,2), C_A{11,2}(:,1), dom_mark_size, cmap_N(5,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=9/low')
plot(C_A{11,2}(:,2), C_A{11,2}(:,1), 'Color', cmap_N(5,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_A{12,2}(:,2), C_A{12,2}(:,1), dom_mark_size, cmap_N(6,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=10/low')
plot(C_A{12,2}(:,2), C_A{12,2}(:,1), 'Color', cmap_N(6,:), 'LineWidth', lw, 'HandleVisibility','off')

scatter(J_unbal_standard_A(2), J_unbal_standard_A(1), 8*dom_mark_size, [0 0 0], 'filled', 'pentagram', 'HandleVisibility','off')
scatter(J_N10_unbal_1_high_A_A(2), J_N10_unbal_1_high_A_A(1), 8*dom_mark_size, [1 0 0], 'filled', 'pentagram', 'MarkerEdgeColor', [0 0 0], 'HandleVisibility','off')
scatter(J_N10_unbal_1_low_A_A(2), J_N10_unbal_1_low_A_A(1), 8*dom_mark_size, [0 0 1], 'filled', 'pentagram', 'MarkerEdgeColor', [0 0 0], 'HandleVisibility','off')
scatter(J_N10_unbal_1_high_D_A(2), J_N10_unbal_1_high_D_A(1), 8*dom_mark_size, [0 1 0], 'filled', 'pentagram', 'MarkerEdgeColor', [0 0 0], 'HandleVisibility','off')
scatter(J_N10_unbal_1_low_D_A(2), J_N10_unbal_1_low_D_A(1), 8*dom_mark_size, [1 0.5 0], 'filled', 'pentagram', 'MarkerEdgeColor', [0 0 0], 'HandleVisibility','off')

box on
xlim([0 70])
ylim([5 12])
fontsize(fig, 20, "points")
lgd = legend('show', 'Location', 'best');
fontsize(lgd, 14, "points")
xlabel('${f}_{\mathrm{time}}$', 'Interpreter','latex', 'FontSize', 30)
ylabel('${f}_{\mathrm{uncertainty}}$', 'Interpreter','latex', 'FontSize', 30)
grid minor
fig.Position = [100, 100, 800, 800];
%%
% Plot pareto optimal frontier
fig = figure(); clf(); 
scatter(C_D{1,2}(:,2), C_D{1,2}(:,1), dom_mark_size, cmap_N(1,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=5/high')
hold on
plot(C_D{1,2}(:,2), C_D{1,2}(:,1), 'Color', cmap_N(1,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_D{2,2}(:,2), C_D{2,2}(:,1), dom_mark_size, cmap_N(2,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=6/high')
plot(C_D{2,2}(:,2), C_D{2,2}(:,1), 'Color', cmap_N(2,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_D{3,2}(:,2), C_D{3,2}(:,1), dom_mark_size, cmap_N(3,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=7/high')
plot(C_D{3,2}(:,2), C_D{3,2}(:,1), 'Color', cmap_N(3,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_D{4,2}(:,2), C_D{4,2}(:,1), dom_mark_size, cmap_N(4,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=8/high')
plot(C_D{4,2}(:,2), C_D{4,2}(:,1), 'Color', cmap_N(4,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_D{5,2}(:,2), C_D{5,2}(:,1), dom_mark_size, cmap_N(5,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=9/high')
plot(C_D{5,2}(:,2), C_D{5,2}(:,1), 'Color', cmap_N(5,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_D{6,2}(:,2), C_D{6,2}(:,1), dom_mark_size, cmap_N(6,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=10/high')
plot(C_D{6,2}(:,2), C_D{6,2}(:,1), 'Color', cmap_N(6,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_D{7,2}(:,2), C_D{7,2}(:,1), dom_mark_size, cmap_N(1,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=5/low')
plot(C_D{7,2}(:,2), C_D{7,2}(:,1), 'Color', cmap_N(1,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_D{8,2}(:,2), C_D{8,2}(:,1), dom_mark_size, cmap_N(2,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=6/low')
plot(C_D{8,2}(:,2), C_D{8,2}(:,1), 'Color', cmap_N(2,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_D{9,2}(:,2), C_D{9,2}(:,1), dom_mark_size, cmap_N(3,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=7/low')
plot(C_D{9,2}(:,2), C_D{9,2}(:,1), 'Color', cmap_N(3,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_D{10,2}(:,2), C_D{10,2}(:,1), dom_mark_size, cmap_N(4,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=8/low')
plot(C_D{10,2}(:,2), C_D{10,2}(:,1), 'Color', cmap_N(4,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_D{11,2}(:,2), C_D{11,2}(:,1), dom_mark_size, cmap_N(5,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=9/low')
plot(C_D{11,2}(:,2), C_D{11,2}(:,1), 'Color', cmap_N(5,:), 'LineWidth', lw, 'HandleVisibility','off')
scatter(C_D{12,2}(:,2), C_D{12,2}(:,1), dom_mark_size, cmap_N(6,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=10/low')
plot(C_D{12,2}(:,2), C_D{12,2}(:,1), 'Color', cmap_N(6,:), 'LineWidth', lw, 'HandleVisibility','off')

scatter(J_unbal_standard_D(2), J_unbal_standard_D(1), 8*dom_mark_size, [0 0 0], 'filled', 'pentagram', 'HandleVisibility','off')
scatter(J_N10_unbal_1_high_A_D(2), J_N10_unbal_1_high_A_D(1), 8*dom_mark_size, [1 0 0], 'filled', 'pentagram', 'MarkerEdgeColor', [0 0 0], 'HandleVisibility','off')
scatter(J_N10_unbal_1_low_A_D(2), J_N10_unbal_1_low_A_D(1), 8*dom_mark_size, [0 0 1], 'filled', 'pentagram', 'MarkerEdgeColor', [0 0 0], 'HandleVisibility','off')
scatter(J_N10_unbal_1_high_D_D(2), J_N10_unbal_1_high_D_D(1), 8*dom_mark_size, [0 1 0], 'filled', 'pentagram', 'MarkerEdgeColor', [0 0 0], 'HandleVisibility','off')
scatter(J_N10_unbal_1_low_D_D(2), J_N10_unbal_1_low_D_D(1), 8*dom_mark_size, [1 0.5 0], 'filled', 'pentagram', 'MarkerEdgeColor', [0 0 0], 'HandleVisibility','off')
yline(J_unbal_standard_A(1), '--', 'LineWidth', 2, 'HandleVisibility','off')
box on
xlim([0 70])
ylim([-8 4])
fontsize(fig, 20, "points")
lgd = legend('show', 'Location', 'best');
fontsize(lgd, 14, "points")
xlabel('${f}_{\mathrm{time}}$', 'Interpreter','latex', 'FontSize', 30)
ylabel('${f}_{\mathrm{uncertainty}}$', 'Interpreter','latex', 'FontSize', 30)
grid minor
fig.Position = [100, 100, 800, 800];

% Plot c_c ranges
fig = figure(); clf(); 
scatter(C_A{1,2}(:,2), C_A{1,3}(:,5), non_mark_size, cmap_N(1,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=5/high')
hold on
scatter(C_A{2,2}(:,2), C_A{2,3}(:,6), non_mark_size, cmap_N(2,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=6/high')
scatter(C_A{3,2}(:,2), C_A{3,3}(:,7), non_mark_size, cmap_N(3,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=7/high')
scatter(C_A{4,2}(:,2), C_A{4,3}(:,8), non_mark_size, cmap_N(4,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=8/high')
scatter(C_A{5,2}(:,2), C_A{5,3}(:,9), non_mark_size, cmap_N(5,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=9/high')
scatter(C_A{6,2}(:,2), C_A{6,3}(:,10), non_mark_size, cmap_N(6,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=10/high')
scatter(C_A{7,2}(:,2), C_A{7,3}(:,5), non_mark_size*1.4, cmap_N(1,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=5/low')
scatter(C_A{8,2}(:,2), C_A{8,3}(:,6), non_mark_size*1.4, cmap_N(2,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=6/low')
scatter(C_A{9,2}(:,2), C_A{9,3}(:,7), non_mark_size*1.4, cmap_N(3,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=7/low')
scatter(C_A{10,2}(:,2), C_A{10,3}(:,8), non_mark_size*1.4, cmap_N(4,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=8/low')
scatter(C_A{11,2}(:,2), C_A{11,3}(:,9), non_mark_size*1.4, cmap_N(5,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=9/low')
scatter(C_A{12,2}(:,2), C_A{12,3}(:,10), non_mark_size*1.4, cmap_N(6,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=10/low')
plot(C_A{1,2}(:,2), C_A{1,3}(:,5), 'Color', cmap_N(1,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_A{2,2}(:,2), C_A{2,3}(:,6), 'Color', cmap_N(2,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_A{3,2}(:,2), C_A{3,3}(:,7), 'Color', cmap_N(3,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_A{4,2}(:,2), C_A{4,3}(:,8), 'Color', cmap_N(4,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_A{5,2}(:,2), C_A{5,3}(:,9), 'Color', cmap_N(5,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_A{6,2}(:,2), C_A{6,3}(:,10), 'Color', cmap_N(6,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_A{7,2}(:,2), C_A{7,3}(:,5), 'Color', cmap_N(1,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_A{8,2}(:,2), C_A{8,3}(:,6), 'Color', cmap_N(2,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_A{9,2}(:,2), C_A{9,3}(:,7), 'Color', cmap_N(3,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_A{10,2}(:,2), C_A{10,3}(:,8), 'Color', cmap_N(4,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_A{11,2}(:,2), C_A{11,3}(:,9), 'Color', cmap_N(5,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_A{12,2}(:,2), C_A{12,3}(:,10), 'Color', cmap_N(6,:), 'LineWidth', lw, 'HandleVisibility','off')
yline(0.4, '--', 'LineWidth', 2, 'HandleVisibility','off')
yline(0.8, '--', 'LineWidth', 2, 'HandleVisibility','off')
box on
ylim([0.35, 0.85])
fontsize(fig, 20, "points")
lgd = legend('show', 'Location', 'best');
fontsize(lgd, 14, "points")
xlabel('${f}_{\mathrm{time}}$', 'Interpreter','latex', 'FontSize', 30)
ylabel('Cathode filling fraction for N-th pulse', 'FontSize', 25)
grid minor
fig.Position = [100, 100, 800, 800];

% Plot c_c ranges
fig = figure(); clf(); 
scatter(C_D{1,2}(:,2), C_D{1,3}(:,5), non_mark_size, cmap_N(1,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=5/high')
hold on
scatter(C_D{2,2}(:,2), C_D{2,3}(:,6), non_mark_size, cmap_N(2,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=6/high')
scatter(C_D{3,2}(:,2), C_D{3,3}(:,7), non_mark_size, cmap_N(3,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=7/high')
scatter(C_D{4,2}(:,2), C_D{4,3}(:,8), non_mark_size, cmap_N(4,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=8/high')
scatter(C_D{5,2}(:,2), C_D{5,3}(:,9), non_mark_size, cmap_N(5,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=9/high')
scatter(C_D{6,2}(:,2), C_D{6,3}(:,10), non_mark_size, cmap_N(6,:), 'o', 'LineWidth', marker_lw, 'DisplayName', 'N=10/high')
scatter(C_D{7,2}(:,2), C_D{7,3}(:,5), non_mark_size*1.4, cmap_N(1,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=5/low')
scatter(C_D{8,2}(:,2), C_D{8,3}(:,6), non_mark_size*1.4, cmap_N(2,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=6/low')
scatter(C_D{9,2}(:,2), C_D{9,3}(:,7), non_mark_size*1.4, cmap_N(3,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=7/low')
scatter(C_D{10,2}(:,2), C_D{10,3}(:,8), non_mark_size*1.4, cmap_N(4,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=8/low')
scatter(C_D{11,2}(:,2), C_D{11,3}(:,9), non_mark_size*1.4, cmap_N(5,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=9/low')
scatter(C_D{12,2}(:,2), C_D{12,3}(:,10), non_mark_size*1.4, cmap_N(6,:), 'x', 'LineWidth', marker_lw, 'DisplayName', 'N=10/low')
plot(C_D{1,2}(:,2), C_D{1,3}(:,5), 'Color', cmap_N(1,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_D{2,2}(:,2), C_D{2,3}(:,6), 'Color', cmap_N(2,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_D{3,2}(:,2), C_D{3,3}(:,7), 'Color', cmap_N(3,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_D{4,2}(:,2), C_D{4,3}(:,8), 'Color', cmap_N(4,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_D{5,2}(:,2), C_D{5,3}(:,9), 'Color', cmap_N(5,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_D{6,2}(:,2), C_D{6,3}(:,10), 'Color', cmap_N(6,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_D{7,2}(:,2), C_D{7,3}(:,5), 'Color', cmap_N(1,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_D{8,2}(:,2), C_D{8,3}(:,6), 'Color', cmap_N(2,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_D{9,2}(:,2), C_D{9,3}(:,7), 'Color', cmap_N(3,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_D{10,2}(:,2), C_D{10,3}(:,8), 'Color', cmap_N(4,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_D{11,2}(:,2), C_D{11,3}(:,9), 'Color', cmap_N(5,:), 'LineWidth', lw, 'HandleVisibility','off')
plot(C_D{12,2}(:,2), C_D{12,3}(:,10), 'Color', cmap_N(6,:), 'LineWidth', lw, 'HandleVisibility','off')
yline(0.4, '--', 'LineWidth', 2, 'HandleVisibility','off')
yline(0.8, '--', 'LineWidth', 2, 'HandleVisibility','off')
box on
ylim([0.35, 0.85])
fontsize(fig, 20, "points")
lgd = legend('show', 'Location', 'best');
fontsize(lgd, 14, "points")
xlabel('${f}_{\mathrm{time}}$', 'Interpreter','latex', 'FontSize', 30)
ylabel('Cathode filling fraction for N-th pulse', 'FontSize', 25)
grid minor
fig.Position = [100, 100, 800, 800];