import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HeaderComponent } from './Component/page/header/header.component';
import { HTTP_INTERCEPTORS, HttpClientModule } from '@angular/common/http';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';
import { HomeComponent } from './Component/page/home/home.component';
import { BrainComponent } from './Component/module/brain/brain.component';
import { AboutusComponent } from './Component/module/aboutus/aboutus.component';
import {MatIconModule} from '@angular/material/icon';
import { MatAutocompleteModule } from '@angular/material/autocomplete';
import { MatInputModule } from '@angular/material/input';
import { BraintumorComponent } from './Component/page/braintumor/braintumor.component';
import { LoginComponent } from './Component/page/login/login.component';
import { TitleComponent } from './Component/module/title/title.component';
import { GoogleLoginProvider, GoogleSigninButtonModule, SocialAuthServiceConfig, SocialLoginModule } from '@abacritt/angularx-social-login';
import { LOGIN_ID, googleLoginOptions } from './model/Login.config';
import { ForgotpasswordComponent } from './Component/module/forgotpassword/forgotpassword.component';
import { NotifyModule } from 'notify-toaster';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { ToastrModule } from 'ngx-toastr';
import { LoadingComponent } from './Component/module/loading/loading.component';
import { loadingInterceptor } from './shared/interceptor/loading.interceptor';
import { userauthInterceptor } from './auth/loginguard/userauth.interceptor';
import { ProfileComponent } from './Component/page/profile/profile.component';
import { BannerComponent } from './Component/module/banner/banner.component';
import { BlogComponent } from './Component/page/blog/blog.component';
import { HomeblogComponent } from './Component/module/homeblog/homeblog.component';
import { ContectComponent } from './Component/module/contect/contect.component';
import { ClipboardModule } from '@angular/cdk/clipboard';
import { ChatbotComponent } from './Component/page/chatbot/chatbot.component';
import { AlluserComponent } from './Component/admin/alluser/alluser.component';
import { MessagesComponent } from './Component/admin/messages/messages.component';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    HomeComponent,
    BrainComponent,
    AboutusComponent,
    BraintumorComponent,
    LoginComponent,
    TitleComponent,
    ForgotpasswordComponent,
    LoadingComponent,
    ProfileComponent,
    BannerComponent,
    BlogComponent,
    HomeblogComponent,
    ContectComponent,
    ChatbotComponent,
    AlluserComponent,
    MessagesComponent,
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    FormsModule,
    ReactiveFormsModule,
    MatIconModule,
    SocialLoginModule,
    GoogleSigninButtonModule,
    NotifyModule,
    BrowserAnimationsModule,
    ToastrModule.forRoot({
      timeOut:4000,
      positionClass:'toast-bottom-right',
      newestOnTop:false
    }),
    MatAutocompleteModule,
    MatInputModule,
    ClipboardModule,
  ],
  providers: [
    provideAnimationsAsync(),
    {provide:HTTP_INTERCEPTORS, useClass:loadingInterceptor,multi:true},
    {provide:HTTP_INTERCEPTORS, useClass:userauthInterceptor,multi:true},
    {
      provide: 'SocialAuthServiceConfig',
      useValue: {
        autoLogin: false,
        providers: [
          {
            id: GoogleLoginProvider.PROVIDER_ID,
            provider: new GoogleLoginProvider(
              LOGIN_ID,
              googleLoginOptions
            )
          },
        ],
        onError: (err) => {
          console.error(err);
        }
      } as SocialAuthServiceConfig,
    },
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
