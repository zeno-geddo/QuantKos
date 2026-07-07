#pragma once
#include <array>


/**
 * @brief Namespace aggregating all tools to compute the exact options prices using semi-analytical formulations.
 */
namespace KOps::Engine::Analytical {
    // 64-points Gauss-Legendre Integrator
    /**
     * @brief Performs 64-point Gauss-Legendre quadrature integration over the interval [a, b].
     *
     * This routine maps the standard [-1, 1] interval to the target range [a, b].
     * It utilizes a symmetric summation of 32 positive nodes and their 32 mirrored counterparts to
     * maintain a compact memory footprint for the weight and root arrays.
     *
     * @tparam IntegrandFunc The callable type (lambda or functor) that satisfies
     *                       the signature `double(double)`.
     * @param integrand A unary callable representing the function to be integrated.
     * @param a The lower bound of the integration interval.
     * @param b The upper bound of the integration interval.
     * @return The numerical approximation of the integral over [a, b].
     */
    template<typename IntegrandFunc>
    inline double integrate_gl64(IntegrandFunc integrand, const double a, const double b) {
        // 64-point Gauss-Legendre roots (nodes) and weights for interval [-1, 1]
        // We only store the 32 positive roots because they are symmetric around 0.
        // From P. Manalastas, 'Computing 32-Place Tables of Zeroes and Weights for Gauss-Legendre Quadrature'
        constexpr std::array<double, 32> gl_roots = {
            0.0243502926634244, 0.0729931217877990, 0.1214628192961206, 0.1696444204239928,
            0.2174236437400071, 0.2646871622087674, 0.3113228719902110, 0.3572201583376681,
            0.4022701579639916, 0.4463660172534641, 0.4894031457070530, 0.5312794640198945,
            0.5718956462026340, 0.6111553551723933, 0.6489654712546573, 0.6852363130542332,
            0.7198818501716108, 0.7528199072605319, 0.7839723589433414, 0.8132653151227976,
            0.8406292962525804, 0.8659993981540928, 0.8893154459951141, 0.9105221370785028,
            0.9295691721319396, 0.9464113748584028, 0.9610087996520537, 0.9733268277899110,
            0.9833362538846260, 0.9910133714767443, 0.9963401167719553, 0.9993050417357721
        };

        constexpr std::array<double, 32> gl_weights = {
            0.0486909570091397, 0.0485754674415034, 0.0483447622348030, 0.0479993885964583,
            0.0475401657148303, 0.0469681828162100, 0.0462847965813144, 0.0454916279274181,
            0.0445905581637566, 0.0435837245293235, 0.0424735151236536, 0.0412625632426235,
            0.0399537411327203, 0.0385501531786156, 0.0370551285402400, 0.0354722132568824,
            0.0338051618371416, 0.0320579283548516, 0.0302346570724025, 0.0283396726142595,
            0.0263774697150547, 0.0243527025687109, 0.0222701738083833, 0.0201348231535302,
            0.0179517157756973, 0.0157260304760247, 0.0134630478967186, 0.0111681394601311,
            0.0088467598263639, 0.0065044579689784, 0.0041470332605625, 0.0017832807216964
        };

        double integral_sum = 0.0;

        // Rescaling the domain : Map [-1, 1] to [a, b]
        // phi(x) = x*((b-a)/2) + (a+b)/2
        // dphi/dx = (b-a) /2
        const double scale = (b - a) / 2.0;
        const double shift = (a + b) / 2.0;

        for (size_t k = 0; k < gl_roots.size(); ++k) {
            const double x_pos = scale * gl_roots[k] + shift; // Positive node
            const double x_neg = scale * (-gl_roots[k]) + shift; // Negative node (Mirrored)
            integral_sum += gl_weights[k] * (integrand(x_pos) + integrand(x_neg));
        }

        // Apply mapping scale (dx to dphi)
        integral_sum *= scale;

        return integral_sum;
    }
}
