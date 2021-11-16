window.addEventListener("load", function () {
    Particles.init({
        selector: ".background",
        connectParticles: true,
        maxParticles: 165,
        color: "#333333",
        retina_detect: true,
        responsive: [
            {
                breakpoint: 1900,
                options: {
                    maxParticles: 125,
                },
            },
            {
                breakpoint: 1600,
                options: {
                    maxParticles: 100,
                },
            },
            {
                breakpoint: 1100,
                options: {
                    maxParticles: 85,
                },
            },
            {
                breakpoint: 850,
                options: {
                    maxParticles: 65,
                },
            },
            {
                breakpoint: 650,
                options: {
                    maxParticles: 50,
                },
            },
            {
                breakpoint: 500,
                options: {
                    maxParticles: 0,
                },
            },
        ],
    });
});