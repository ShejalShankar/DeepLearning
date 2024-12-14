def generate_samples(checkpoint_path, latent_dim=100, num_samples=1):
    G = Generator3D(latent_dim=latent_dim).to(device)
    G.load_state_dict(torch.load(checkpoint_path, map_location=device))
    G.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        fake = G(z).cpu().numpy()  # shape: (num_samples,1,16,16,16)

    for i in range(num_samples):
        voxel_volume = fake[i, 0]
        print(f"Generated sample {i} shape:", voxel_volume.shape)
        visualize_voxel(voxel_volume, threshold=0.5, colormap='viridis)
generate_samples("/content/checkpoints/generator_epoch_4.pth", latent_dim=100, num_samples=1)
