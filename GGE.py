class GGE(nn.Module):
    def __init__(
        self,
        num_iters=10,
        step_size=0.5,
        lambda_region=1.0,
        lambda_curv=0.2,
        eps=1e-6,
    ):
        super().__init__()
        self.T = num_iters
        self.eta = step_size
        self.lambda_r = lambda_region
        self.lambda_c = lambda_curv
        self.eps = eps

    def init_level_set(self, mask):
        # mask: [B,1,H,W] in [0,1]
        return 2 * mask - 1

    def curvature(self, phi):
        dy, dx = torch.gradient(phi, dim=(2, 3))
        norm = torch.sqrt(dx**2 + dy**2 + self.eps)
        nx, ny = dx / norm, dy / norm
        nxx = torch.gradient(nx, dim=3)[0]
        nyy = torch.gradient(ny, dim=2)[0]
        return nxx + nyy

    def forward(self, image, coarse_mask, gt=None):
        """
        image: [B,1,H,W]
        coarse_mask: [B,1,H,W]
        """
        phi = self.init_level_set(coarse_mask)

        for _ in range(self.T):
            H = torch.sigmoid(phi)

            c1 = (H * image).sum() / (H.sum() + self.eps)
            c2 = ((1 - H) * image).sum() / ((1 - H).sum() + self.eps)

            region_force = (image - c1)**2 - (image - c2)**2
            curv = self.curvature(phi)

            phi = phi - self.eta * (
                self.lambda_r * region_force
                - self.lambda_c * curv
            )

        refined = torch.sigmoid(phi)

        if gt is None:
            return refined

        loss = dice_loss(refined, gt)
        return refined, loss
