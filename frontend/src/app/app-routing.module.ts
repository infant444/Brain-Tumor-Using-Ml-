import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './Component/page/home/home.component';
import { BraintumorComponent } from './Component/page/braintumor/braintumor.component';
import { LoginComponent } from './Component/page/login/login.component';
import { ForgotpasswordComponent } from './Component/module/forgotpassword/forgotpassword.component';
import { LoginAuthGuard } from './auth/loginguard/loginauth.guard';
import { ProfileComponent } from './Component/page/profile/profile.component';
import { BrainComponent } from './Component/module/brain/brain.component';
import { AlluserComponent } from './Component/admin/alluser/alluser.component';
import { AdminauthGuard } from './auth/adminguard/adminauth.guard';
import { MessagesComponent } from './Component/admin/messages/messages.component';

const routes: Routes = [
  {path:"",component:HomeComponent},
  {path:"braintumor",component:BraintumorComponent,canActivate:[LoginAuthGuard]},
  {path:"login",component:LoginComponent},
  {path:"login/forgotpassword",component:ForgotpasswordComponent},
  {path:"profile",component:ProfileComponent,canActivate:[LoginAuthGuard]},
  {path:"braintumor/:reportid",component:BraintumorComponent,canActivate:[LoginAuthGuard]},
  {path:"report/:reportid",component:BrainComponent,canActivate:[LoginAuthGuard]},
  {path:"admin/alluser",component:AlluserComponent},
  {path:"user/:id",component:ProfileComponent},
  {path:"admin/message",component:MessagesComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
