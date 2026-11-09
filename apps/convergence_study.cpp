#include <iomanip>
#include <iostream>
#include "Fem1D.h"
#include "Thermoelastic1D.h"
#include <cmath>
using std::cout;
using std::endl;

struct ChainErrors {
    double tip = 0.0;      // errore sullo spostamento della punta (barra con l'estremo destro libero)
    double stress = 0.0;   // errore massimo sulla tensione (barra con l'estremo destro bloccato)
};

ThermoMechanicalProblem sine_problem(std::size_t N, double dt, std::size_t steps, double theta,
                                     RightEnd right_end) {
    const double pi = std::acos(-1.0);
    ThermoMechanicalProblem p;
    p.x = uniform_mesh(1.0, N);
    p.T0.resize(p.x.size());
    for (std::size_t i = 0; i < p.x.size(); ++i) {
        p.T0[i] = std::sin(pi * p.x[i]);
    }
    p.rho_c = 1.0;
    p.k = 1.0;
    p.q = 0.0;
    p.T_left = 0.0;
    p.T_right = 0.0;
    p.dt = dt;
    p.n_steps = steps;
    p.theta = theta;
    p.E = 3.0;
    p.alpha = 2.0;
    p.T_ref = 0.0;
    p.right_end = right_end;
    return p;
}

void print_chain_row(std::size_t level, double step, const ChainErrors& errors, const ChainErrors& previous) {
    std::cout << "| " << std::setw(5) << level << " | " << std::scientific << std::setprecision(2) << step
              << " | " << std::setw(9) << errors.tip << " | ";
    if (previous.tip > 0.0) {
        std::cout << std::fixed << std::setprecision(2) << std::setw(5) << observed_order(previous.tip, errors.tip, 2.0);
    } else {
        std::cout << std::setw(5) << "-";
    }
    std::cout << " | " << std::scientific << std::setprecision(2) << std::setw(12) << errors.stress << " |\n";
}

void print_chain_header(const char* level_name, const char* step_name) {
    std::cout << "| " << std::setw(5) << level_name << " | " << std::setw(8) << step_name
              << " | tip error | order | stress error |\n";
    std::cout << "|------:|---------:|----------:|------:|-------------:|\n";
}

ChainErrors chain_errors(std::size_t N, double dt, std::size_t steps, double theta) {

    ChainErrors errors;
    const double pi = std::acos(-1.0);
    double t = dt * steps;

    ThermoMechanicalProblem problem_free = sine_problem(N, dt, steps, theta, RightEnd::Free);
    ThermoMechanicalResult result_free=  solve_thermomechanical(problem_free);
    double final_u = result_free.u.back();
    errors.tip = std::fabs(final_u - 2*problem_free.alpha*(std::exp(-pi*pi*t))/pi);

    ThermoMechanicalProblem problem_fixed = sine_problem(N, dt, steps, theta, RightEnd::Fixed);
    ThermoMechanicalResult result_fixed =  solve_thermomechanical(problem_fixed);
    double exact_solution_fixed = -2*problem_fixed.E*problem_fixed.alpha*(std::exp(-pi*pi*t))/pi;
    double error = 0.0;
    for(const auto &i:result_fixed.sigma){
        if(std::fabs(i - exact_solution_fixed) > error){
            error = std::fabs(i - exact_solution_fixed);}
}
    errors.stress = error;

    return errors;
}

int main() {

    cout << "### Spatial convergence (Crank-Nicolson, dt = 1e-4, t = 0.1)" << endl << endl;
    cout << "|     N |        h | L2 error | order |" << endl;
    cout << "|------:|---------:|---------:|------:|" << endl;

    size_t mesh[5] = {10, 20, 40, 80, 160};
    double previous_error = 0.0;
    for (const auto &i : mesh) {
        double h = 1.0 / i;
        double error = decaying_sine_error(i, 1e-4, 1000, 0.5);

        cout << "| " << std::setw(5) << i << " | " << std::scientific << std::setprecision(2) << h << " | " << error << " | ";
        if (i == 10) {
            cout << std::setw(5) << "-";
        } else {
            cout << std::fixed << std::setprecision(2) << std::setw(5) << observed_order(previous_error, error, 2.0);
        }
        cout << " |" << endl;

        previous_error = error;
    }

    cout << endl;
    cout << "### Temporal convergence (backward Euler, N = 800, t = 0.1)" << endl;
    cout << "| steps |       dt | L2 error | order |" << endl;
    cout << "|------:|---------:|---------:|------:|" << endl;
    size_t temporal_steps[4] = {5, 10, 20, 40};
    previous_error = 0.0;
    for (const auto &i : temporal_steps) {
        double dt = 0.1 / i;
        double error = decaying_sine_error(800, dt, i, 1);

        cout << "| " << std::setw(5) << i << " | " << std::scientific << std::setprecision(2) << dt << " | " << error << " | ";
        if (i == 5) {
            cout << std::setw(5) << "-";
        } else {
            cout << std::fixed << std::setprecision(2) << std::setw(5) << observed_order(previous_error, error, 2.0);
        }
        cout << " |" << endl;

        previous_error = error;
    }
    
    cout << endl;
    cout << "### Temporal convergence (Crank-Nicolson, N = 800, t = 0.1)" << endl;
    cout << "| steps |       dt | L2 error | order |" << endl;
    cout << "|------:|---------:|---------:|------:|" << endl;

    previous_error = 0.0;
    for (const auto &i : temporal_steps) {
        double dt = 0.1 / i;
        double error = decaying_sine_error(800, dt, i, 0.5);

        cout << "| " << std::setw(5) << i << " | " << std::scientific << std::setprecision(2) << dt << " | " << error << " | ";
        if (i == 5) {
            cout << std::setw(5) << "-";
        } else {
            cout << std::fixed << std::setprecision(2) << std::setw(5) << observed_order(previous_error, error, 2.0);
        }
        cout << " |" << endl;

        previous_error = error;
    }
    
    std::cout << "\n### Chain: spatial convergence (Crank-Nicolson, dt = 1e-4, t = 0.1)\n\n";
    print_chain_header("N", "h");
    ChainErrors chain_previous;
    for (std::size_t N : {10, 20, 40, 80, 160}) {
        const ChainErrors errors = chain_errors(N, 1e-4, 1000, 0.5);
        print_chain_row(N, 1.0 / N, errors, chain_previous);
        chain_previous = errors;
    }

    for (double theta : {1.0, 0.5}) {
        std::cout << "\n### Chain: temporal convergence (" << (theta == 1.0 ? "backward Euler" : "Crank-Nicolson")
                  << ", N = 800, t = 0.1)\n\n";
        print_chain_header("steps", "dt");
        chain_previous = ChainErrors();
        for (std::size_t steps : {5, 10, 20, 40}) {
            const double dt = 0.1 / steps;
            const ChainErrors errors = chain_errors(800, dt, steps, theta);
            print_chain_row(steps, dt, errors, chain_previous);
            chain_previous = errors;
        }
    }

    return 0;
}