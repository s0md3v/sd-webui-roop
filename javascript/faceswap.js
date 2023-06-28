window.onbeforeunload = function() {
    // Prevent the stable diffusion window from being closed by mistake
    return "Are you sure ?";
};
