elif case == "lowerT":  # upper factor of H⁻¹ as the transpose of its lower factor
    torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H))).mT.contiguous()