import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { User } from '../../model/user';
import { BehaviorSubject, Observable, tap } from 'rxjs';
import { CHANGEPASSWORD, FEEDBACK, LOGIN, LOGIN_GOOGLE, SENDOTP, SIGNUP } from '../../model/url';
import { NotifyService } from 'notify-toaster';
import { ToastrService } from 'ngx-toastr';
import { USER_KEY } from '../../model/Login.config';
import { Contect } from '../../model/contect';

@Injectable({
  providedIn: 'root'
})
export class UserService {
  private userSubject = new BehaviorSubject<User>(this.getUserLocalStorage());
  public userObeservable: Observable<User>;
  constructor(private http: HttpClient, private toaste: ToastrService) {
    this.userObeservable = this.userSubject.asObservable();

  }

  Login(user: User): Observable<User> {
    return this.http.post<User>(LOGIN, user).pipe(
      tap({
        next: (user) => {
          this.userSubject.next(user)
          this.toaste.success("Signin Sucessfully!..")
          this.setUserToLocalStorage(user)

        },
        error: (err) => {
          this.toaste.error(err.error.toString())
        }
      })
    )
  }
  Signup(user: User): Observable<User> {
    return this.http.post<User>(SIGNUP, user).pipe(
      tap({
        next: (user) => {
          // console.log(user)
          this.userSubject.next(user)
          this.toaste.success("signup Sucessfully!..")
          this.setUserToLocalStorage(user)

        },
        error: (err) => {
          this.toaste.error(err.error.toString())
        }
      })
    )
  }
  Goolge_Login(user: User): Observable<User> {
    return this.http.post<User>(LOGIN_GOOGLE, user).pipe(
      tap({
        next: (user) => {
          this.userSubject.next(user)
          this.toaste.success("login Sucessfully")
          this.setUserToLocalStorage(user)

        },
        error: (err) => {
          this.toaste.error(err.error.toString())
        }
      })
    )
  }
  logout(){
    this.userSubject.next(new User());
    localStorage.removeItem(USER_KEY);
    window.location.reload();
  }

  sendfeedback(x:Contect){
    return this.http.post(FEEDBACK,x);
  }

  sendopt(email:string):Observable<any>{
    return this.http.get<any>(SENDOTP+email);
  }
  changepassword(email:string,x:any):Observable<User>{
    return this.http.post<User>(CHANGEPASSWORD+email,x)
  }
  private setUserToLocalStorage(user: User) {
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  }




  private getUserLocalStorage(): User {
    const x = localStorage.getItem(USER_KEY);
    if (x) {
      return JSON.parse(x);
    }
    else {
      return new User();
    }
  }

  public get currentUser(): User {
    return this.userSubject.value;
  }
}
