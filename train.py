def train_3d_gan(dataloader, epochs=5, latent_dim=100, lr=0.0002, checkpoint_dir="/content/checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    G = Generator3D(latent_dim=latent_dim).to(device)
    D = Discriminator3D().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optim_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (real_batch, _) in enumerate(dataloader):
            real_batch = real_batch.to(device)  # shape: (N,1,16,16,16)
            N = real_batch.size(0)
            # Train Discriminator
            D.zero_grad()

            labels_real = torch.ones(N, device=device)
            labels_fake = torch.zeros(N, device=device)

            pred_real = D(real_batch)
            loss_D_real = criterion(pred_real, labels_real)

            z = torch.randn(N, latent_dim, device=device)
            fake = G(z)
            pred_fake = D(fake.detach())
            loss_D_fake = criterion(pred_fake, labels_fake)

            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            optim_D.step()

            # Train Generator
            G.zero_grad()
            pred_fake_for_G = D(fake)
            loss_G = criterion(pred_fake_for_G, labels_real)
            loss_G.backward()
            optim_G.step()

            if i % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}] Step [{i}/{len(dataloader)}] "
                      f"D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")

        # Save checkpoints each epoch
        torch.save(G.state_dict(), os.path.join(checkpoint_dir, f"generator_epoch_{epoch}.pth"))
        torch.save(D.state_dict(), os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch}.pth"))
        print(f"Saved checkpoint for epoch {epoch}")

    return G, D
